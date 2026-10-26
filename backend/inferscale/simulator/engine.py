"""Core discrete-event simulation engine for LLM inference serving."""

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterator
from enum import Enum
import math

from inferscale.schemas.request import RequestSchema
from inferscale.config import settings


class EventType(Enum):
    """Types of events in the simulation."""
    ARRIVAL = "arrival"
    START_PREFILL = "start_prefill"
    END_PREFILL = "end_prefill"
    START_DECODE = "start_decode"
    END_DECODE = "end_decode"
    REPLICA_READY = "replica_ready"
    SCALE_ACTION = "scale_action"
    DECISION_POINT = "decision_point"
    EPISODE_END = "episode_end"


class ReplicaState(Enum):
    """States of a replica."""
    STARTING = "starting"
    ACTIVE = "active"
    DRAINING = "draining"
    STOPPED = "stopped"


@dataclass
class Replica:
    """Represents a single inference replica."""
    id: int
    state: ReplicaState = ReplicaState.STOPPED
    start_time: float = 0.0
    ready_time: float = 0.0
    active_requests: int = 0
    total_processed: int = 0
    total_tokens_processed: int = 0
    in_flight: List[str] = field(default_factory=list)


@dataclass
class Request:
    """Represents a single request in the simulation."""
    id: str
    arrival_time: float
    prompt_tokens: int
    output_tokens: int
    state: str = "waiting"
    queue_entry_time: float = 0.0
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0
    decode_start_time: float = 0.0
    decode_end_time: float = 0.0
    assigned_replica: Optional[int] = None
    sla_violated: bool = False
    source: Optional[str] = None


@dataclass
class SimulationEvent:
    """Event in the discrete-event simulation."""
    time: float
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    sequence: int = 0

    def __lt__(self, other: "SimulationEvent") -> bool:
        if self.time != other.time:
            return self.time < other.time
        return (self.priority, self.sequence) < (other.priority, other.sequence)


class ServiceTimeModel:
    """Analytical service time model for LLM inference."""

    def __init__(
        self,
        prefill_coeff: float = 0.001,
        decode_coeff: float = 0.0005,
        batch_overhead: float = 0.0001,
    ):
        self.prefill_coeff = prefill_coeff
        self.decode_coeff = decode_coeff
        self.batch_overhead = batch_overhead

    def prefill_time(self, prompt_tokens: int, batch_size: int = 1) -> float:
        """Compute prefill time for a batch of requests."""
        return self.prefill_coeff * prompt_tokens * batch_size + self.batch_overhead * batch_size

    def decode_time(self, output_tokens: int, batch_size: int = 1) -> float:
        """Compute decode time for a batch of requests."""
        return self.decode_coeff * output_tokens * batch_size

    def total_time(self, prompt_tokens: int, output_tokens: int, batch_size: int = 1) -> float:
        """Compute total service time."""
        return self.prefill_time(prompt_tokens, batch_size) + self.decode_time(output_tokens, batch_size)


class SimulatorEngine:
    """Discrete-event simulation engine for LLM inference autoscaling."""

    def __init__(
        self,
        requests: List[RequestSchema],
        initial_replicas: int = 2,
        min_replicas: int = 1,
        max_replicas: int = 8,
        startup_delay: float = 10.0,
        cooldown: float = 30.0,
        decision_interval: float = 5.0,
        episode_duration: float = 3600.0,
        sla_threshold: float = 30.0,
        replica_capacity: int = 10,
        seed: int = 42,
        cost_per_replica_second: Optional[float] = None,
        slo_target: float = 0.99,
    ):
        self.requests = sorted(requests, key=lambda r: r.arrival_time_seconds)
        self.initial_replicas = initial_replicas
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.startup_delay = startup_delay
        self.cooldown = cooldown
        self.decision_interval = decision_interval
        self.episode_duration = episode_duration
        self.sla_threshold = sla_threshold
        self.replica_capacity = replica_capacity
        self.seed = seed
        self.cost_per_replica_second = (
            settings.gpu_cost_per_replica_second
            if cost_per_replica_second is None else cost_per_replica_second
        )
        if not math.isfinite(self.cost_per_replica_second) or self.cost_per_replica_second < 0:
            raise ValueError("cost_per_replica_second must be finite and nonnegative")
        if not math.isfinite(slo_target) or not 0 <= slo_target <= 1:
            raise ValueError("slo_target must be between zero and one")
        self.slo_target = slo_target
        self._queue_seconds = 0.0

        # Initialize random state
        self.rng = random.Random(seed)

        # Simulation state
        self.current_time = 0.0
        self.replicas: Dict[int, Replica] = {}
        self.waiting_queue: List[Request] = []
        self.completed_requests: List[Request] = []
        self.requests_by_id: Dict[str, Request] = {}  # All requests indexed by ID
        self.events: List[SimulationEvent] = []
        self.next_replica_id = 0
        self.next_event_id = 0
        self.last_scale_time = -float("inf")
        self.pending_scale_up = 0
        self.pending_scale_down = 0
        self.decision_points: List[float] = []
        self.action_history: List[Dict] = []
        self.reward_history: List[float] = []
        self.is_running = False
        self.is_paused = False

        # Service time model
        self.service_model = ServiceTimeModel()

        # Metrics
        self.metrics = {
            "total_replica_seconds": 0.0,
            "total_cost": 0.0,
            "accounting_version": 2,
            "starting_replica_seconds": 0.0,
            "active_replica_seconds": 0.0,
            "draining_replica_seconds": 0.0,
            "arrivals": 0,
            "completions": 0,
            "sla_violations": 0,
            "max_queue_depth": 0,
            "queue_depths": [],
            "latencies": [],
            "queueing_delays": [],
            "service_times": [],
            "throughput": [],
            "goodput": [],
            "scaling_events": 0,
            "burst_recovery_times": [],
        }

        # Initialize replicas
        self._initialize_replicas()

        # Schedule arrivals
        self._schedule_arrivals()

        # Schedule episode end
        self._schedule_event(self.episode_duration, EventType.EPISODE_END, priority=100)

        # Schedule first decision point
        self._schedule_event(0, EventType.DECISION_POINT)

    def _initialize_replicas(self) -> None:
        """Initialize the initial set of replicas."""
        for i in range(self.initial_replicas):
            replica = Replica(id=self.next_replica_id)
            self.next_replica_id += 1
            self._start_replica(replica)

    def _start_replica(self, replica: Replica) -> None:
        """Start a replica (it will be ready after startup_delay)."""
        replica.state = ReplicaState.STARTING
        replica.start_time = self.current_time
        replica.ready_time = self.current_time + self.startup_delay
        self.replicas[replica.id] = replica
        self._schedule_event(replica.ready_time, EventType.REPLICA_READY, {"replica_id": replica.id})

    def _schedule_arrivals(self) -> None:
        """Schedule all request arrivals."""
        for req_schema in self.requests:
            # An episode observes arrivals in [0, horizon), and completions through the horizon.
            if not 0 <= req_schema.arrival_time_seconds < self.episode_duration:
                continue
            req = Request(
                id=req_schema.id,
                arrival_time=req_schema.arrival_time_seconds,
                prompt_tokens=req_schema.prompt_tokens,
                output_tokens=req_schema.output_tokens,
                source=req_schema.source,
            )
            self._schedule_event(req.arrival_time, EventType.ARRIVAL, {"request": req})

    def _schedule_event(self, time: float, event_type: EventType, data: Optional[Dict] = None, priority: int = 0) -> None:
        """Schedule a new event."""
        if time > self.episode_duration:
            return
        event = SimulationEvent(
            time=time,
            event_type=event_type,
            data=data or {},
            priority=priority,
            sequence=self.next_event_id,
        )
        self.next_event_id += 1
        heapq.heappush(self.events, event)

    def _advance_time(self, time: float) -> None:
        """Integrate billable state durations before changing any replica state."""
        elapsed = time - self.current_time
        if elapsed < 0:
            raise ValueError("Simulation time cannot move backwards")
        for state in (ReplicaState.STARTING, ReplicaState.ACTIVE, ReplicaState.DRAINING):
            count = sum(replica.state == state for replica in self.replicas.values())
            self.metrics[f"{state.value}_replica_seconds"] += count * elapsed
        self.metrics["total_replica_seconds"] = sum(
            self.metrics[f"{state.value}_replica_seconds"]
            for state in (ReplicaState.STARTING, ReplicaState.ACTIVE, ReplicaState.DRAINING)
        )
        self.metrics["total_cost"] = (
            self.metrics["total_replica_seconds"] * self.cost_per_replica_second
        )
        self._queue_seconds += len(self.waiting_queue) * elapsed
        self.current_time = time

    def step(self) -> Optional[Dict]:
        """Execute one simulation step."""
        if not self.events or self.current_time >= self.episode_duration:
            # Callers may loop on the queue; discard events that cannot run after the horizon.
            self.events.clear()
            return None

        event = heapq.heappop(self.events)
        self._advance_time(event.time)
        result = self._process_event(event)
        # Finish all events at the horizon, including exact-deadline completions,
        # before a caller's current_time < horizon loop exits.
        if self.current_time == self.episode_duration:
            while self.events and self.events[0].time == self.episode_duration:
                extra = self._process_event(heapq.heappop(self.events))
                if extra is not None:
                    result = extra
            self.events.clear()
        return result

    def _process_event(self, event: SimulationEvent) -> Optional[Dict]:

        # Process event
        if event.event_type == EventType.ARRIVAL:
            self._handle_arrival(event.data["request"])
        elif event.event_type == EventType.REPLICA_READY:
            self._handle_replica_ready(event.data["replica_id"])
        elif event.event_type == EventType.START_PREFILL:
            self._handle_start_prefill(event.data["replica_id"], event.data["request_id"])
        elif event.event_type == EventType.END_PREFILL:
            self._handle_end_prefill(event.data["replica_id"], event.data["request_id"])
        elif event.event_type == EventType.START_DECODE:
            self._handle_start_decode(event.data["replica_id"], event.data["request_id"])
        elif event.event_type == EventType.END_DECODE:
            self._handle_end_decode(event.data["replica_id"], event.data["request_id"])
        elif event.event_type == EventType.DECISION_POINT:
            return self._handle_decision_point()
        elif event.event_type == EventType.SCALE_ACTION:
            self._handle_scale_action(event.data["action"])
        elif event.event_type == EventType.EPISODE_END:
            return self._handle_episode_end()

        return None

    def _handle_arrival(self, request: Request) -> None:
        """Handle a request arrival."""
        request.queue_entry_time = self.current_time
        request.state = "waiting"
        self.waiting_queue.append(request)
        self.requests_by_id[request.id] = request
        self.metrics["arrivals"] += 1
        self._update_queue_metrics()
        self._try_schedule_requests()

    def _handle_replica_ready(self, replica_id: int) -> None:
        """Handle a replica becoming ready."""
        replica = self.replicas.get(replica_id)
        if replica:
            replica.state = ReplicaState.ACTIVE
            self.pending_scale_up = max(0, self.pending_scale_up - 1)
            self._try_schedule_requests()

    def _handle_start_prefill(self, replica_id: int, request_id: str) -> None:
        """Handle start of prefill."""
        replica = self.replicas.get(replica_id)
        request = self._find_request(request_id)
        if replica and request:
            request.state = "prefill"
            request.prefill_start_time = self.current_time
            request.assigned_replica = replica_id
            replica.active_requests += 1
            replica.in_flight.append(request_id)

    def _handle_end_prefill(self, replica_id: int, request_id: str) -> None:
        """Handle end of prefill."""
        request = self._find_request(request_id)
        if request:
            request.state = "decoding"
            request.prefill_end_time = self.current_time
            self._schedule_event(self.current_time, EventType.START_DECODE, {"replica_id": replica_id, "request_id": request_id})

    def _handle_start_decode(self, replica_id: int, request_id: str) -> None:
        """Handle start of decode."""
        request = self._find_request(request_id)
        if request:
            request.decode_start_time = self.current_time
            decode_time = self.service_model.decode_time(request.output_tokens)
            self._schedule_event(self.current_time + decode_time, EventType.END_DECODE, {"replica_id": replica_id, "request_id": request_id})

    def _handle_end_decode(self, replica_id: int, request_id: str) -> None:
        """Handle end of decode (request completion)."""
        request = self._find_request(request_id)
        replica = self.replicas.get(replica_id)
        if request and replica:
            request.state = "completed"
            request.decode_end_time = self.current_time
            request.sla_violated = (request.decode_end_time - request.arrival_time) > self.sla_threshold
            if request.sla_violated:
                self.metrics["sla_violations"] += 1

            replica.active_requests -= 1
            replica.in_flight.remove(request_id)
            if replica.state == ReplicaState.DRAINING and replica.active_requests == 0:
                replica.state = ReplicaState.STOPPED
                self.pending_scale_down = max(0, self.pending_scale_down - 1)
            replica.total_processed += 1
            replica.total_tokens_processed += request.prompt_tokens + request.output_tokens
            self.completed_requests.append(request)
            self.waiting_queue = [r for r in self.waiting_queue if r.id != request_id]
            self.metrics["completions"] += 1
            self._update_latency_metrics(request)
            self._try_schedule_requests()

    def _try_schedule_requests(self) -> None:
        """Try to schedule waiting requests to available replicas."""
        active_replicas = [r for r in self.replicas.values() if r.state == ReplicaState.ACTIVE]
        for replica in active_replicas:
            while self.waiting_queue and replica.active_requests < self.replica_capacity:
                request = self.waiting_queue.pop(0)
                self._schedule_request_on_replica(replica, request)

    def _schedule_request_on_replica(self, replica: Replica, request: Request) -> None:
        """Schedule a request on a replica."""
        request.state = "prefill"
        request.prefill_start_time = self.current_time
        request.assigned_replica = replica.id
        replica.active_requests += 1
        replica.in_flight.append(request.id)
        prefill_time = self.service_model.prefill_time(request.prompt_tokens)
        self._schedule_event(self.current_time + prefill_time, EventType.END_PREFILL, {"replica_id": replica.id, "request_id": request.id})

    def _find_request(self, request_id: str) -> Optional[Request]:
        """Find a request by ID."""
        return self.requests_by_id.get(request_id)

    def _handle_decision_point(self) -> Dict:
        """Handle a decision point - returns observation for policy."""
        self.decision_points.append(self.current_time)
        # Direct policy evaluation needs a decision throughout the episode, not only at t=0.
        next_time = self.current_time + self.decision_interval
        if self.decision_interval > 0 and next_time < self.episode_duration:
            self._schedule_event(next_time, EventType.DECISION_POINT)
        obs = self.get_observation()
        return {
            "time": self.current_time,
            "observation": obs,
            "decision_point": True,
        }

    def _handle_scale_action(self, action: int) -> None:
        """Handle a scaling action from policy."""
        # 0: scale down, 1: hold, 2: scale up
        active = len([r for r in self.replicas.values() if r.state == ReplicaState.ACTIVE])
        starting = len([r for r in self.replicas.values() if r.state == ReplicaState.STARTING])
        total = active + starting

        # Check cooldown
        if action != 1 and self.current_time - self.last_scale_time < self.cooldown:
            self.action_history.append({"time": self.current_time, "action": "hold", "replicas": total})
            return

        if action == 2 and total < self.max_replicas:
            # Scale up
            replica = Replica(id=self.next_replica_id)
            self.next_replica_id += 1
            self._start_replica(replica)
            self.pending_scale_up += 1
            self.metrics["scaling_events"] += 1
            self.action_history.append({"time": self.current_time, "action": "scale_up", "replicas": total + 1})
            self.last_scale_time = self.current_time
        elif action == 0 and total > self.min_replicas:
            # Scale down - find a replica to drain
            active_replicas = [r for r in self.replicas.values() if r.state == ReplicaState.ACTIVE]
            if active_replicas:
                # Drain the one with least active requests
                target = min(active_replicas, key=lambda r: r.active_requests)
                target.state = ReplicaState.DRAINING
                self.pending_scale_down += 1
                if target.active_requests == 0:
                    target.state = ReplicaState.STOPPED
                    self.pending_scale_down -= 1
                self.metrics["scaling_events"] += 1
                self.action_history.append({"time": self.current_time, "action": "scale_down", "replicas": total - 1})
                self.last_scale_time = self.current_time
            else:
                self.action_history.append({"time": self.current_time, "action": "hold", "replicas": total})
        else:
            self.action_history.append({"time": self.current_time, "action": "hold", "replicas": total})

    def _handle_episode_end(self) -> Dict:
        """Handle episode termination."""
        self.is_running = False
        return {
            "time": self.current_time,
            "episode_end": True,
            "metrics": self.get_metrics(),
        }

    def get_current_replicas(self) -> int:
        """Get current total replicas (active + starting)."""
        active = len([r for r in self.replicas.values() if r.state == ReplicaState.ACTIVE])
        starting = len([r for r in self.replicas.values() if r.state == ReplicaState.STARTING])
        return active + starting

    def get_observation(self) -> Dict[str, float]:
        """Get current observation for policy."""
        active = len([r for r in self.replicas.values() if r.state == ReplicaState.ACTIVE])
        starting = len([r for r in self.replicas.values() if r.state == ReplicaState.STARTING])
        total = active + starting
        queue_len = len(self.waiting_queue)
        estimated_work = sum(r.prompt_tokens + r.output_tokens for r in self.waiting_queue)

        # Recent arrival rate (last 60s)
        recent_arrivals = [r for r in self.completed_requests if self.current_time - r.arrival_time <= 60]
        arrival_rate = len(recent_arrivals) / 60.0 if recent_arrivals else 0.0

        # Recent prompt/output stats
        recent_prompts = [r.prompt_tokens for r in recent_arrivals] if recent_arrivals else [0]
        recent_outputs = [r.output_tokens for r in recent_arrivals] if recent_arrivals else [0]

        # Rolling p95 latency
        recent_latencies = [r.decode_end_time - r.arrival_time for r in self.completed_requests[-100:] if r.decode_end_time]
        p95_latency = sorted(recent_latencies)[int(len(recent_latencies) * 0.95)] if recent_latencies else 0.0

        # SLA violation rate
        sla_rate = self.metrics["sla_violations"] / max(1, self.metrics["completions"])

        # Utilization
        utilization = active / max(1, self.max_replicas)

        # Cooldown state
        cooldown_remaining = max(0.0, self.last_scale_time + self.cooldown - self.current_time)

        return {
            "queue_length": queue_len,
            "estimated_work": estimated_work,
            "arrival_rate": arrival_rate,
            "avg_prompt_tokens": sum(recent_prompts) / len(recent_prompts),
            "avg_output_tokens": sum(recent_outputs) / len(recent_outputs),
            "active_replicas": active,
            "total_replicas": total,
            "utilization": utilization,
            "p95_latency": p95_latency,
            "sla_violation_rate": sla_rate,
            "cooldown_remaining": cooldown_remaining,
            "starting_replicas": starting,
        }

    def take_action(self, action: int) -> None:
        """Apply a scaling action."""
        self._schedule_event(self.current_time, EventType.SCALE_ACTION, {"action": action})

    def get_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        completed = self.completed_requests
        latencies = sorted(r.decode_end_time - r.arrival_time for r in completed)
        queueing_delays = [r.prefill_start_time - r.arrival_time for r in completed]
        service_times = [r.decode_end_time - r.prefill_start_time for r in completed]
        unfinished = [r for r in self.requests_by_id.values() if r.state != "completed"]
        overdue = sum(
            r.arrival_time + self.sla_threshold < self.current_time for r in unfinished
        )
        pending = len(unfinished) - overdue
        late = self.metrics["sla_violations"]
        on_time = len(completed) - late
        arrivals = self.metrics["arrivals"]
        evaluated = len(completed) + overdue
        lower = on_time / arrivals if arrivals else None
        upper = (on_time + pending) / arrivals if arrivals else None
        slo_met = None
        if arrivals:
            if lower >= self.slo_target:
                slo_met = True
            elif upper < self.slo_target:
                slo_met = False
        elapsed = self.current_time

        return {
            **self.metrics,
            "cost_per_replica_second": self.cost_per_replica_second,
            **{
                f"{state}_cost": self.metrics[f"{state}_replica_seconds"]
                * self.cost_per_replica_second
                for state in ("starting", "active", "draining")
            },
            "p50_latency_ms": sorted(latencies)[len(latencies)//2] * 1000 if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)] * 1000 if latencies else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)] * 1000 if latencies else 0,
            "avg_queueing_delay_ms": sum(queueing_delays) / len(queueing_delays) * 1000 if queueing_delays else 0,
            "avg_service_time_ms": sum(service_times) / len(service_times) * 1000 if service_times else 0,
            "goodput": on_time / elapsed if elapsed else 0.0,
            "throughput": len(completed) / elapsed if elapsed else 0.0,
            "avg_active_replicas": (
                self.metrics["active_replica_seconds"] / elapsed if elapsed else 0.0
            ),
            "avg_queue_depth": self._queue_seconds / elapsed if elapsed else 0.0,
            "unfinished_requests": len(unfinished),
            "queued_requests": len(self.waiting_queue),
            "in_flight_requests": len(unfinished) - len(self.waiting_queue),
            "on_time_completions": on_time,
            "late_completions": late,
            "overdue_requests": overdue,
            "pending_requests": pending,
            "slo_violations": late + overdue,
            "slo_evaluated_requests": evaluated,
            "sla_violation_rate": (late + overdue) / evaluated if evaluated else None,
            "completed_request_sla_violation_rate": late / len(completed) if completed else None,
            "completion_rate": len(completed) / arrivals if arrivals else None,
            "slo_attainment_lower_bound": lower,
            "slo_attainment_upper_bound": upper,
            "slo_target": self.slo_target,
            "slo_met": slo_met,
            "cost_per_on_time_request": self.metrics["total_cost"] / on_time if on_time else None,
        }

    def _update_queue_metrics(self) -> None:
        """Update queue depth metrics."""
        self.metrics["queue_depths"].append(len(self.waiting_queue))
        self.metrics["max_queue_depth"] = max(self.metrics["max_queue_depth"], len(self.waiting_queue))

    def _update_latency_metrics(self, request: Request) -> None:
        """Update latency metrics for a completed request."""
        latency = request.decode_end_time - request.arrival_time
        queueing = request.prefill_start_time - request.arrival_time
        service = request.decode_end_time - request.prefill_start_time
        self.metrics["latencies"].append(latency)
        self.metrics["queueing_delays"].append(queueing)
        self.metrics["service_times"].append(service)

    def run_until(self, time: float) -> List[Dict]:
        """Run simulation until a specific time."""
        self.is_running = True
        results = []
        target = min(time, self.episode_duration)
        while (
            self.events and self.events[0].time <= target
            and self.current_time < self.episode_duration
        ):
            result = self.step()
            if result:
                results.append(result)
        if target > self.current_time:
            self._advance_time(target)
        return results

    def run(self) -> List[Dict]:
        """Run simulation to completion."""
        return self.run_until(self.episode_duration)

    def get_timeline(self) -> List[Dict]:
        """Get timeline of all events for visualization."""
        timeline = []
        for r in self.completed_requests:
            timeline.append({
                "type": "request",
                "id": r.id,
                "arrival": r.arrival_time,
                "queue_entry": r.queue_entry_time,
                "prefill_start": r.prefill_start_time,
                "prefill_end": r.prefill_end_time,
                "decode_start": r.decode_start_time,
                "decode_end": r.decode_end_time,
                "replica": r.assigned_replica,
                "sla_violated": r.sla_violated,
            })
        for a in self.action_history:
            timeline.append({
                "type": "action",
                "time": a["time"],
                "action": a["action"],
                "replicas": a["replicas"],
            })
        return timeline

    def reset(self) -> None:
        """Reset the simulator to initial state."""
        self.__init__(
            requests=self.requests,
            initial_replicas=self.initial_replicas,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            startup_delay=self.startup_delay,
            cooldown=self.cooldown,
            decision_interval=self.decision_interval,
            episode_duration=self.episode_duration,
            sla_threshold=self.sla_threshold,
            replica_capacity=self.replica_capacity,
            seed=self.seed,
            cost_per_replica_second=self.cost_per_replica_second,
            slo_target=self.slo_target,
        )


def run_simulation(
    requests: List[RequestSchema],
    policy: str = "threshold",
    config: Optional[Dict] = None,
) -> Tuple[Dict, List[Dict]]:
    """Convenience function to run a simulation with a policy."""
    from inferscale.policies import get_policy

    cfg = config or {}
    engine = SimulatorEngine(
        requests=requests,
        initial_replicas=cfg.get("initial_replicas", 2),
        min_replicas=cfg.get("min_replicas", 1),
        max_replicas=cfg.get("max_replicas", 8),
        startup_delay=cfg.get("startup_delay", 10.0),
        cooldown=cfg.get("cooldown", 30.0),
        decision_interval=cfg.get("decision_interval", 5.0),
        episode_duration=cfg.get("episode_duration", 3600.0),
        sla_threshold=cfg.get("sla_threshold", 30.0),
        replica_capacity=cfg.get("replica_capacity", 10),
        seed=cfg.get("random_seed", 42),
        cost_per_replica_second=cfg.get("cost_per_replica_second"),
        slo_target=cfg.get("slo_target", 0.99),
    )

    policy_obj = get_policy(policy, cfg)

    events = []
    while engine.events and engine.current_time < engine.episode_duration:
        result = engine.step()
        if result and result.get("decision_point"):
            obs = result["observation"]
            action = policy_obj.act(obs, engine)
            engine.take_action(action)
            events.append({"time": engine.current_time, "action": action, "observation": obs})

    return engine.get_metrics(), engine.get_timeline()
