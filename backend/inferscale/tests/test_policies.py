"""Tests for baseline policies."""

import pytest

from inferscale.policies.fixed import FixedCapacityPolicy
from inferscale.policies.threshold import ThresholdPolicy
from inferscale.policies.queue import QueueBasedPolicy
from inferscale.policies.ppo import PPOPolicy
from inferscale.simulator.engine import SimulatorEngine
from inferscale.schemas.request import RequestSchema


@pytest.fixture
def sample_requests():
    requests = []
    for i in range(20):
        requests.append(RequestSchema(
            id=f"req_{i}",
            arrival_time_seconds=i * 0.5,
            prompt_tokens=50,
            output_tokens=30,
            source="test",
        ))
    return requests


@pytest.fixture
def engine(sample_requests):
    return SimulatorEngine(
        requests=sample_requests,
        initial_replicas=2,
        min_replicas=1,
        max_replicas=4,
        startup_delay=5.0,
        cooldown=10.0,
        decision_interval=2.0,
        episode_duration=20.0,
        sla_threshold=5.0,
        replica_capacity=5,
        seed=42,
    )


@pytest.fixture
def observation(engine):
    """Get a sample observation."""
    # Run a few steps to get meaningful state
    for _ in range(3):
        engine.step()
    return engine.get_observation()


class TestFixedCapacityPolicy:
    """Tests for fixed capacity policy."""

    def test_scale_up_when_below_capacity(self, engine):
        policy = FixedCapacityPolicy({"fixed_capacity": 4})
        obs = {"active_replicas": 1, "starting_replicas": 0, "time": 0}
        
        action = policy.act(obs, engine)
        assert action == 2  # scale up

    def test_scale_down_when_above_capacity(self, engine):
        policy = FixedCapacityPolicy({"fixed_capacity": 2})
        obs = {"active_replicas": 3, "starting_replicas": 0, "time": 0}
        
        action = policy.act(obs, engine)
        assert action == 0  # scale down

    def test_hold_when_at_capacity(self, engine):
        policy = FixedCapacityPolicy({"fixed_capacity": 2})
        obs = {"active_replicas": 2, "starting_replicas": 0, "time": 0}
        
        action = policy.act(obs, engine)
        assert action == 1  # hold

    def test_respects_max_replicas(self, engine):
        engine.max_replicas = 3
        policy = FixedCapacityPolicy({"fixed_capacity": 5})
        obs = {"active_replicas": 3, "starting_replicas": 0, "time": 0}
        
        action = policy.act(obs, engine)
        # Should not scale up beyond max
        assert action != 2


class TestThresholdPolicy:
    """Tests for threshold-based autoscaling policy."""

    def test_scale_up_on_high_utilization(self, engine):
        policy = ThresholdPolicy({"threshold_upper": 0.7, "threshold_lower": 0.3, "cooldown": 0})
        obs = {
            "active_replicas": 2,
            "starting_replicas": 0,
            "utilization": 0.9,
            "queue_length": 0,
            "time": 100,
        }
        
        action = policy.act(obs, engine)
        assert action == 2  # scale up

    def test_scale_up_on_queue_pressure(self, engine):
        policy = ThresholdPolicy({"threshold_upper": 0.7, "threshold_lower": 0.3, "cooldown": 0})
        obs = {
            "active_replicas": 2,
            "starting_replicas": 0,
            "utilization": 0.5,
            "queue_length": 10,  # High queue pressure
            "time": 100,
        }
        
        action = policy.act(obs, engine)
        assert action == 2  # scale up

    def test_scale_down_after_sustained_low_utilization(self, engine):
        policy = ThresholdPolicy({
            "threshold_upper": 0.7,
            "threshold_lower": 0.3,
            "scaling_in_duration": 5.0,
            "cooldown": 0
        })
        
        # First call - utilization low, starts timer
        obs = {"active_replicas": 3, "starting_replicas": 0, "utilization": 0.1, "queue_length": 0, "time": 100}
        action = policy.act(obs, engine)
        assert action == 1  # hold (timer started)
        
        # Second call - still low but not enough time passed
        obs["time"] = 102
        action = policy.act(obs, engine)
        assert action == 1  # hold
        
        # Third call - enough time passed
        obs["time"] = 110
        action = policy.act(obs, engine)
        assert action == 0  # scale down

    def test_resets_timer_on_utilization_increase(self, engine):
        policy = ThresholdPolicy({
            "threshold_upper": 0.7,
            "threshold_lower": 0.3,
            "scaling_in_duration": 5.0,
            "cooldown": 0
        })
        
        obs = {"active_replicas": 3, "starting_replicas": 0, "utilization": 0.1, "queue_length": 0, "time": 100}
        policy.act(obs, engine)
        
        obs["time"] = 102
        obs["utilization"] = 0.5  # Above lower threshold
        action = policy.act(obs, engine)
        assert action == 1
        
        # Timer should be reset, need to wait again
        obs["time"] = 104
        obs["utilization"] = 0.1
        action = policy.act(obs, engine)
        assert action == 1  # hold (timer restarted)

    def test_respects_cooldown(self, engine):
        policy = ThresholdPolicy({"threshold_upper": 0.5, "cooldown": 10.0})
        
        obs = {"active_replicas": 1, "starting_replicas": 0, "utilization": 0.9, "queue_length": 0, "time": 100}
        action = policy.act(obs, engine)
        assert action == 2  # scale up
        
        # Immediate second call should be blocked by cooldown
        obs["time"] = 101
        obs["active_replicas"] = 2
        obs["utilization"] = 0.9
        action = policy.act(obs, engine)
        assert action == 1  # hold due to cooldown
        
        # After cooldown
        obs["time"] = 115
        action = policy.act(obs, engine)
        assert action == 2  # can scale up again


class TestQueueBasedPolicy:
    """Tests for queue-based autoscaling policy."""

    def test_scale_up_on_high_work_per_replica(self, engine):
        policy = QueueBasedPolicy({"queue_scale_up": 5.0, "cooldown": 0})
        obs = {
            "active_replicas": 2,
            "starting_replicas": 0,
            "queue_length": 20,
            "estimated_work": 2000,  # 100 tokens * 20 requests
            "time": 100,
        }
        
        action = policy.act(obs, engine)
        assert action == 2  # scale up

    def test_scale_down_on_low_work_per_replica(self, engine):
        policy = QueueBasedPolicy({"queue_scale_down": 1.0, "cooldown": 0})
        obs = {
            "active_replicas": 4,
            "starting_replicas": 0,
            "queue_length": 1,
            "estimated_work": 2,  # work_per_replica = 0.5 < 1.0
            "time": 100,
        }
        
        action = policy.act(obs, engine)
        assert action == 0  # scale down

    def test_hold_when_in_range(self, engine):
        policy = QueueBasedPolicy({"queue_scale_up": 10.0, "queue_scale_down": 1.0, "cooldown": 0})
        obs = {
            "active_replicas": 2,
            "starting_replicas": 0,
            "queue_length": 5,
            "estimated_work": 10,  # work_per_replica = 5, between 1 and 10
            "time": 100,
        }
        
        action = policy.act(obs, engine)
        assert action == 1  # hold

    def test_longer_cooldown_for_scale_down(self, engine):
        policy = QueueBasedPolicy({"queue_scale_down": 1.0, "cooldown": 10.0})
        
        obs = {"active_replicas": 3, "starting_replicas": 0, "queue_length": 1, "estimated_work": 1, "time": 100}
        action = policy.act(obs, engine)
        assert action == 0  # scale down
        
        # Immediate second scale down should be blocked by 2x cooldown
        obs["time"] = 105
        action = policy.act(obs, engine)
        assert action == 1  # hold


class TestPPOPolicy:
    """Tests for PPO policy."""

    def test_fallback_behavior_when_no_model(self, engine):
        policy = PPOPolicy({})  # No model path
        
        # High utilization -> scale up
        obs = {"active_replicas": 1, "starting_replicas": 0, "utilization": 0.9, "time": 0}
        action = policy.act(obs, engine)
        assert action == 2
        
        # Low utilization -> scale down
        obs = {"active_replicas": 3, "starting_replicas": 0, "utilization": 0.1, "time": 0}
        action = policy.act(obs, engine)
        assert action == 0
        
        # Medium utilization -> hold
        obs = {"active_replicas": 2, "starting_replicas": 0, "utilization": 0.5, "time": 0}
        action = policy.act(obs, engine)
        assert action == 1

    def test_observation_to_array(self, engine):
        policy = PPOPolicy({})
        obs = {
            "queue_length": 10,
            "estimated_work": 1000,
            "arrival_rate": 5.0,
            "avg_prompt_tokens": 50,
            "avg_output_tokens": 30,
            "active_replicas": 2,
            "total_replicas": 3,
            "utilization": 0.6,
            "p95_latency": 100,
            "sla_violation_rate": 0.1,
            "cooldown_remaining": 0,
            "starting_replicas": 1,
        }
        
        arr = policy._observation_to_array(obs)
        assert arr.shape == (12,)
        assert arr[0] == 10  # queue_length
        assert arr[1] == 1000  # estimated_work


class TestPolicyIntegration:
    """Integration tests running policies with simulator."""

    def test_all_policies_complete_simulation(self, sample_requests):
        """All policies should complete simulation without errors."""
        for policy_name in ["fixed", "threshold", "queue"]:
            engine = SimulatorEngine(
                requests=sample_requests,
                initial_replicas=2,
                min_replicas=1,
                max_replicas=4,
                startup_delay=5.0,
                cooldown=10.0,
                decision_interval=2.0,
                episode_duration=20.0,
                sla_threshold=5.0,
                replica_capacity=5,
                seed=42,
            )
            
            from inferscale.policies import get_policy
            policy = get_policy(policy_name, {"fixed_capacity": 3})
            
            step_count = 0
            while engine.events and engine.current_time < 20.0:
                result = engine.step()
                step_count += 1
                
                if result and result.get("decision_point"):
                    obs = result["observation"]
                    obs["time"] = engine.current_time
                    action = policy.act(obs, engine)
                    engine.take_action(action)
                
                if step_count > 1000:  # Safety limit
                    break
            
            metrics = engine.get_metrics()
            assert metrics["completions"] > 0
            assert metrics["arrivals"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])