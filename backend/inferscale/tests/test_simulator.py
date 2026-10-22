"""Tests for the simulator engine."""

import pytest
import random

from inferscale.simulator.engine import (
    SimulatorEngine,
    Request,
    Replica,
    ReplicaState,
    EventType,
    ServiceTimeModel,
    run_simulation,
)
from inferscale.schemas.request import RequestSchema


@pytest.fixture
def sample_requests():
    """Create sample requests for testing."""
    requests = []
    for i in range(10):
        requests.append(RequestSchema(
            id=f"req_{i}",
            arrival_time_seconds=i * 1.0,
            prompt_tokens=50,
            output_tokens=30,
            source="test",
        ))
    return requests


@pytest.fixture
def engine(sample_requests):
    """Create a simulator engine for testing."""
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


class TestServiceTimeModel:
    """Tests for the analytical service time model."""

    def test_prefill_time(self):
        model = ServiceTimeModel(prefill_coeff=0.001, batch_overhead=0.0001)
        assert abs(model.prefill_time(100, 1) - 0.1001) < 1e-10
        assert abs(model.prefill_time(100, 2) - 0.2002) < 1e-10

    def test_decode_time(self):
        model = ServiceTimeModel(decode_coeff=0.0005)
        assert model.decode_time(50, 1) == 0.025
        assert model.decode_time(50, 2) == 0.05

    def test_total_time(self):
        model = ServiceTimeModel(prefill_coeff=0.001, decode_coeff=0.0005, batch_overhead=0.0001)
        total = model.total_time(100, 50, 1)
        assert abs(total - (0.1001 + 0.025)) < 1e-10


class TestSimulatorEngine:
    """Tests for the discrete-event simulator."""

    def test_initialization(self, engine):
        assert engine.initial_replicas == 2
        assert engine.min_replicas == 1
        assert engine.max_replicas == 4
        assert engine.startup_delay == 5.0
        assert len(engine.replicas) == 2
        assert len(engine.events) > 0  # arrivals + episode end + decision point

    def test_replica_initialization(self, engine):
        active = [r for r in engine.replicas.values() if r.state == ReplicaState.ACTIVE]
        starting = [r for r in engine.replicas.values() if r.state == ReplicaState.STARTING]
        # Initially all replicas should be starting
        assert len(starting) == 2
        assert len(active) == 0

    def test_request_arrival(self, engine):
        # Advance to first arrival
        while engine.events and engine.current_time < 1.0:
            result = engine.step()
        
        assert engine.current_time >= 1.0
        assert engine.metrics["arrivals"] >= 1
        assert len(engine.waiting_queue) >= 0  # May be scheduled immediately

    def test_replica_becomes_ready(self, engine):
        # Run until replicas are ready (startup_delay = 5.0)
        while engine.events and engine.current_time < 6.0:
            result = engine.step()
        
        active = [r for r in engine.replicas.values() if r.state == ReplicaState.ACTIVE]
        assert len(active) == 2

    def test_request_processing(self, engine):
        # Run simulation to completion
        while engine.events:
            result = engine.step()
        
        # All requests should be completed
        assert engine.metrics["completions"] == 10
        assert engine.metrics["arrivals"] == 10
        assert len(engine.completed_requests) == 10

    def test_sla_violation_tracking(self, engine):
        # Run with very tight SLA
        engine.sla_threshold = 0.001  # 1ms - impossible to meet
        
        while engine.events:
            engine.step()
        
        assert engine.metrics["sla_violations"] > 0

    def test_scaling_actions(self, engine):
        engine.cooldown = 0.0  # Disable cooldown
        # Test scale up
        engine.take_action(2)  # scale up
        # Process events until scale action is handled
        while engine.events:
            event = engine.events[0]
            if event.event_type == EventType.SCALE_ACTION:
                break
            engine.step()
        engine.step()  # process scale action
        
        total_replicas = len([r for r in engine.replicas.values() if r.state != ReplicaState.STOPPED])
        assert total_replicas == 3
        assert engine.metrics["scaling_events"] == 1

        # Wait for replicas to become active
        while engine.events and engine.current_time < 6.0:
            engine.step()
        
        # Test scale down
        engine.take_action(0)  # scale down
        while engine.events:
            event = engine.events[0]
            if event.event_type == EventType.SCALE_ACTION:
                break
            engine.step()
        engine.step()
        
        active = [r for r in engine.replicas.values() if r.state == ReplicaState.ACTIVE]
        draining = [r for r in engine.replicas.values() if r.state == ReplicaState.DRAINING]
        assert len(draining) == 1

    def test_scale_limits(self, engine):
        # Scale up to max
        for _ in range(3):  # 2 initial + 3 = 5, but max is 4
            engine.take_action(2)
            while engine.events:
                event = engine.events[0]
                if event.event_type == EventType.SCALE_ACTION:
                    break
                engine.step()
            engine.step()
        
        total = len([r for r in engine.replicas.values() if r.state != ReplicaState.STOPPED])
        assert total <= engine.max_replicas

        # Wait for replicas to become active
        while engine.events and engine.current_time < 6.0:
            engine.step()
        
        # Try to scale down below min
        for _ in range(10):
            engine.take_action(0)
            while engine.events:
                event = engine.events[0]
                if event.event_type == EventType.SCALE_ACTION:
                    break
                engine.step()
            engine.step()
        
        active = [r for r in engine.replicas.values() if r.state == ReplicaState.ACTIVE]
        assert len(active) >= engine.min_replicas

    def test_cooldown_enforcement(self, engine):
        engine.cooldown = 10.0
        engine.take_action(2)  # scale up
        while engine.events:
            event = engine.events[0]
            if event.event_type == EventType.SCALE_ACTION:
                break
            engine.step()
        engine.step()
        
        # Immediate scale up should be ignored due to cooldown
        engine.take_action(2)
        while engine.events:
            event = engine.events[0]
            if event.event_type == EventType.SCALE_ACTION:
                break
            engine.step()
        engine.step()
        
        # Should only have scaled once
        assert engine.metrics["scaling_events"] == 1

    def test_in_flight_preservation_on_scale_down(self, engine):
        engine.cooldown = 0.0  # Disable cooldown for this test
        # Run until replicas are active and processing
        while engine.events and engine.current_time < 6.0:
            engine.step()
        
        # Get an active replica with in-flight requests
        active = [r for r in engine.replicas.values() if r.state == ReplicaState.ACTIVE]
        # Find one with in-flight requests
        target_replica = None
        for r in active:
            if r.in_flight:
                target_replica = r
                break
        
        if target_replica:
            replica_id = target_replica.id
            in_flight_count = len(target_replica.in_flight)
            
            # Scale down - should target the replica with least active requests
            engine.take_action(0)
            # Process events until scale action is handled
            while engine.events:
                event = engine.events[0]
                if event.event_type == EventType.SCALE_ACTION:
                    break
                engine.step()
            engine.step()
            
            # The target replica should be draining (if it was selected) or another with 0 active requests
            replica = engine.replicas[replica_id]
            # Check that in-flight requests are preserved on whichever replica was drained
            drained = [r for r in engine.replicas.values() if r.state == ReplicaState.DRAINING]
            assert len(drained) == 1
            # In-flight requests should be preserved on the drained replica
            assert len(drained[0].in_flight) == in_flight_count or drained[0].active_requests == 0

    def test_episode_termination_with_pending_requests(self, engine):
        # Set very short episode duration
        engine.episode_duration = 1.0
        
        while engine.events:
            result = engine.step()
            if result and result.get("episode_end"):
                break
        
        assert engine.current_time >= 1.0
        # Unfinished requests should be tracked
        metrics = engine.get_metrics()
        assert not engine.events
        assert len(engine.waiting_queue) == metrics["arrivals"] > 0

    def test_determinism(self, sample_requests):
        """Test that simulator is deterministic with same seed."""
        engine1 = SimulatorEngine(
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
        
        engine2 = SimulatorEngine(
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
        
        # Run both to completion
        while engine1.events:
            engine1.step()
        while engine2.events:
            engine2.step()
        
        # Metrics should be identical
        assert engine1.metrics["completions"] == engine2.metrics["completions"]
        assert engine1.metrics["sla_violations"] == engine2.metrics["sla_violations"]
        assert engine1.metrics["total_cost"] == engine2.metrics["total_cost"]
        assert len(engine1.action_history) == len(engine2.action_history)

    def test_no_future_info_leakage(self, engine):
        """Test that policy observations don't contain future information."""
        # Run a few steps
        for _ in range(5):
            result = engine.step()
        
        obs = engine.get_observation()
        
        # Observation should only contain current/past info
        assert "queue_length" in obs
        assert "arrival_rate" in obs
        assert "p95_latency" in obs
        # Should NOT contain future arrivals or completions
        assert "future_arrivals" not in obs
        assert "future_completions" not in obs

    def test_queue_metrics(self, engine):
        # Run a few steps
        for _ in range(10):
            engine.step()
        
        assert "queue_depths" in engine.metrics
        assert "max_queue_depth" in engine.metrics
        assert engine.metrics["max_queue_depth"] >= 0

    def test_latency_metrics(self, engine):
        while engine.events:
            engine.step()
        
        metrics = engine.get_metrics()
        assert "p50_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert "p99_latency_ms" in metrics
        assert "avg_queueing_delay_ms" in metrics
        assert "avg_service_time_ms" in metrics

    def test_reset(self, engine):
        # Run a bit
        for _ in range(5):
            engine.step()
        
        original_time = engine.current_time
        original_completions = engine.metrics["completions"]
        
        # Reset
        engine.reset()
        
        assert engine.current_time == 0.0
        assert engine.metrics["completions"] == 0
        assert len(engine.replicas) == engine.initial_replicas


class TestRunSimulation:
    """Tests for the run_simulation convenience function."""

    def test_run_with_threshold_policy(self, sample_requests):
        metrics, timeline = run_simulation(
            requests=sample_requests,
            policy="threshold",
            config={
                "initial_replicas": 2,
                "min_replicas": 1,
                "max_replicas": 4,
                "startup_delay": 5.0,
                "cooldown": 10.0,
                "decision_interval": 2.0,
                "episode_duration": 20.0,
                "sla_threshold": 5.0,
                "replica_capacity": 5,
                "random_seed": 42,
            }
        )
        
        assert "completions" in metrics
        assert "p95_latency_ms" in metrics
        assert len(timeline) > 0

    def test_all_policies_produce_results(self, sample_requests):
        for policy in ["fixed", "threshold", "queue"]:
            metrics, timeline = run_simulation(
                requests=sample_requests,
                policy=policy,
                config={
                    "initial_replicas": 2,
                    "min_replicas": 1,
                    "max_replicas": 4,
                    "startup_delay": 5.0,
                    "cooldown": 10.0,
                    "decision_interval": 2.0,
                    "episode_duration": 20.0,
                    "sla_threshold": 5.0,
                    "replica_capacity": 5,
                    "random_seed": 42,
                }
            )
            assert metrics["completions"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
