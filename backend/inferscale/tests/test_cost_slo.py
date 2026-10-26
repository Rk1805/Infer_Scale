"""Analytical accounting examples and deadline edge cases with hand-computed results."""

import pytest

from inferscale.rl.env import InferScaleEnv
from inferscale.schemas.experiment import ExperimentConfig
from inferscale.schemas.request import RequestSchema
from inferscale.simulator.engine import ReplicaState, ServiceTimeModel, SimulatorEngine


def request(identifier, arrival=0, duration=1):
    return RequestSchema(id=identifier, arrival_time_seconds=arrival,
                         prompt_tokens=duration, output_tokens=1)


def engine(requests=None, **kwargs):
    config = dict(initial_replicas=1, max_replicas=4, startup_delay=0, cooldown=0,
                  decision_interval=2, episode_duration=10, cost_per_replica_second=0.5)
    config.update(kwargs)
    simulation = SimulatorEngine(requests=requests or [], **config)
    simulation.service_model = ServiceTimeModel(prefill_coeff=1, decode_coeff=0, batch_overhead=0)
    return simulation


def test_four_idle_replicas_cost_2400_replica_seconds():
    simulation = engine(initial_replicas=4, episode_duration=600, startup_delay=10)
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["total_replica_seconds"] == 2400
    assert metrics["starting_replica_seconds"] == 40
    assert metrics["active_replica_seconds"] == 2360
    assert metrics["total_cost"] == 1200
    assert metrics["avg_active_replicas"] == pytest.approx(2360 / 600)


def test_startup_beyond_horizon_is_not_prepaid():
    simulation = engine(initial_replicas=2, startup_delay=20, episode_duration=5)
    assert simulation.get_metrics()["total_cost"] == 0
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["starting_replica_seconds"] == 10
    assert metrics["active_cost"] == 0
    assert metrics["total_cost"] == 5


def test_scale_up_and_idle_scale_down_are_billed_until_stop():
    simulation = engine(initial_replicas=2)
    simulation.run_until(4)
    simulation.take_action(0)
    simulation.run_until(4)
    assert sum(r.state == ReplicaState.STOPPED for r in simulation.replicas.values()) == 1
    simulation.run()
    assert simulation.get_metrics()["total_replica_seconds"] == 14
    assert simulation.get_metrics()["total_cost"] == 7

    simulation = engine(startup_delay=3)
    simulation.run_until(4)
    simulation.take_action(2)
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["total_replica_seconds"] == 16
    assert metrics["starting_replica_seconds"] == 6
    assert metrics["active_replica_seconds"] == 10


def test_busy_replica_drains_then_stops_billing():
    simulation = engine([request("a", duration=3), request("b", duration=3)],
                        initial_replicas=2, replica_capacity=1)
    simulation.run_until(1)
    simulation.take_action(0)
    simulation.run_until(2)
    assert sum(r.state == ReplicaState.DRAINING for r in simulation.replicas.values()) == 1
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["completions"] == 2
    assert metrics["draining_replica_seconds"] == 2
    assert metrics["active_replica_seconds"] == 11
    assert metrics["total_cost"] == 6.5
    assert metrics["starting_cost"] + metrics["active_cost"] + metrics["draining_cost"] == 6.5
    assert simulation.pending_scale_down == simulation.pending_scale_up == 0


def test_partial_interval_does_not_overshoot_and_reading_does_not_bill_twice():
    simulation = engine(startup_delay=5)
    simulation.run_until(0.25)
    first = simulation.get_metrics()
    assert simulation.current_time == 0.25
    assert first["total_cost"] == 0.125
    assert simulation.get_metrics() == first
    simulation.run()
    assert simulation.get_metrics()["total_cost"] == 5
    simulation.reset()
    assert simulation.get_metrics()["total_cost"] == 0
    assert simulation.cost_per_replica_second == 0.5


@pytest.mark.parametrize("price", [-1, float("inf"), float("nan")])
def test_invalid_prices_rejected(price):
    with pytest.raises(ValueError):
        engine(cost_per_replica_second=price)
    with pytest.raises(ValueError):
        ExperimentConfig(cost_per_replica_second=price)


@pytest.mark.parametrize("target", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_slo_targets_rejected(target):
    with pytest.raises(ValueError):
        engine(slo_target=target)
    with pytest.raises(ValueError):
        ExperimentConfig(slo_target=target)


def test_empty_workload_has_cost_but_no_slo_evidence():
    simulation = engine()
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["total_cost"] == 5
    assert metrics["sla_violation_rate"] is None
    assert metrics["slo_attainment_lower_bound"] is None
    assert metrics["slo_met"] is None
    assert metrics["cost_per_on_time_request"] is None


def test_no_completions_cannot_hide_overdue_queue():
    simulation = engine([request("a"), request("b", arrival=9)], startup_delay=20,
                        sla_threshold=3)
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["completions"] == 0
    assert metrics["queued_requests"] == 2
    assert metrics["overdue_requests"] == 1
    assert metrics["pending_requests"] == 1
    assert metrics["sla_violation_rate"] == 1
    assert metrics["completed_request_sla_violation_rate"] is None
    assert metrics["slo_met"] is False
    assert metrics["avg_queue_depth"] == pytest.approx(1.1)


@pytest.mark.parametrize("target,expected", [(0.75, False), (0.4, None), (0.2, True)])
def test_mixed_outcomes_bound_attainment_instead_of_discarding_unfinished(target, expected):
    simulation = engine([
        request("on-time"), request("late", duration=5), request("overdue", duration=20),
        request("pending", arrival=9, duration=20),
    ], sla_threshold=3, slo_target=target)
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["arrivals"] == 4
    assert metrics["on_time_completions"] == metrics["late_completions"] == 1
    assert metrics["overdue_requests"] == metrics["pending_requests"] == 1
    assert metrics["queued_requests"] == 0 and metrics["in_flight_requests"] == 2
    assert metrics["slo_attainment_lower_bound"] == 0.25
    assert metrics["slo_attainment_upper_bound"] == 0.5
    assert metrics["sla_violation_rate"] == pytest.approx(2 / 3)
    assert metrics["completed_request_sla_violation_rate"] == 0.5
    assert metrics["slo_met"] is expected


def test_completion_at_deadline_and_horizon_counts_as_on_time():
    simulation = engine([request("exact", duration=5), request("outside", arrival=5)],
                        episode_duration=5, sla_threshold=5, slo_target=1)
    simulation.run()
    metrics = simulation.get_metrics()
    assert metrics["arrivals"] == metrics["completions"] == 1
    assert metrics["slo_met"] is True
    assert metrics["unfinished_requests"] == metrics["slo_violations"] == 0


def test_unfinished_exactly_at_deadline_is_pending_then_overdue():
    simulation = engine([request("a", duration=20)], sla_threshold=3)
    simulation.run_until(3)
    assert simulation.get_metrics()["pending_requests"] == 1
    simulation.run_until(3.01)
    assert simulation.get_metrics()["overdue_requests"] == 1


def test_rl_reward_uses_each_interval_and_does_not_repenalize_old_violations():
    env = InferScaleEnv(requests=[request("late", duration=100000)], config={
        "initial_replicas": 1, "max_replicas": 4, "startup_delay": 20,
        "decision_interval": 2, "episode_duration": 6, "sla_threshold": 3,
        "cost_per_replica_second": 0.5,
    })
    env.reset()
    infos = [env.step(1)[4] for _ in range(3)]
    assert [info["interval_metrics"]["cost"] for info in infos] == [1, 1, 1]
    assert [info["interval_metrics"]["new_slo_violations"] for info in infos] == [0, 1, 0]
    assert [info["reward_components"]["cost"] for info in infos] == pytest.approx([-0.025] * 3)
    assert infos[1]["reward_components"]["sla"] == -1
    assert infos[2]["reward_components"]["sla"] == 0


def test_zero_price_rewards_are_finite():
    env = InferScaleEnv(config={"cost_per_replica_second": 0, "episode_duration": 10})
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert info["reward_components"]["cost"] == 0
    assert info["interval_metrics"]["cost"] == 0
    assert reward <= 0


def test_experiment_runner_simulates_traffic_and_propagates_billing_settings():
    from inferscale.experiments.runner import ExperimentRunner, JobState

    runner = ExperimentRunner()
    job = runner.create_job(ExperimentConfig(
        episode_duration=20, replica_startup_delay=1, cost_per_replica_second=0.25,
        slo_target=0.9, random_seed=12,
    ))
    runner._run_job(job)
    assert job.state == JobState.COMPLETED, job.error
    assert job.results["arrivals"] > 0
    assert job.results["completions"] > 0
    assert job.results["slo_target"] == 0.9
    assert job.results["total_cost"] == pytest.approx(job.results["total_replica_seconds"] * 0.25)


def test_scaling_reward_penalizes_actual_actions_once():
    env = InferScaleEnv(config={"initial_replicas": 1, "max_replicas": 3,
                                "cooldown": 30, "episode_duration": 10})
    env.reset()
    first = env.step(2)[4]
    second = env.step(2)[4]
    assert first["interval_metrics"]["scaling_events"] == 1
    assert first["reward_components"]["churn"] < 0
    assert second["interval_metrics"]["scaling_events"] == 0
    assert second["reward_components"]["churn"] == 0
