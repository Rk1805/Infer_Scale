"""Command-line interface for InferScale."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from inferscale.config import settings
from inferscale.data.synth import SYNTHETIC_WORKLOADS, generate_steady_workload
from inferscale.data.loader import load_requests
from inferscale.simulator.engine import SimulatorEngine, run_simulation
from inferscale.policies import get_policy
from inferscale.rl.env import InferScaleEnv
from inferscale.experiments.persistence import db


def generate_data(args):
    """Generate synthetic dataset."""
    print(f"Generating {args.workload_type} workload...")
    
    if args.workload_type == "mixed":
        requests = []
        for wt in ["steady", "periodic", "flash_crowd"]:
            requests.extend(SYNTHETIC_WORKLOADS[wt](num_requests=args.num_requests // 3))
    else:
        gen_func = SYNTHETIC_WORKLOADS.get(args.workload_type, SYNTHETIC_WORKLOADS["steady"])
        requests = gen_func(num_requests=args.num_requests)
    
    # Save to CSV
    import csv
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "arrival_time_seconds", "prompt_tokens", "output_tokens", "source"])
        for req in requests:
            writer.writerow([req.id, req.arrival_time_seconds, req.prompt_tokens, req.output_tokens, req.source or ""])
    
    print(f"Generated {len(requests)} requests to {args.output}")


def run_sim(args):
    """Run a simulation with a specific policy."""
    print(f"Running simulation with {args.policy} policy...")
    
    # Load requests
    if args.dataset == "synthetic":
        if args.workload_type == "mixed":
            requests = []
            for wt in ["steady", "periodic", "flash_crowd"]:
                requests.extend(SYNTHETIC_WORKLOADS[wt](num_requests=1000))
        else:
            gen_func = SYNTHETIC_WORKLOADS.get(args.workload_type, SYNTHETIC_WORKLOADS["steady"])
            requests = gen_func()
    else:
        requests = load_requests(args.dataset)
    
    # Apply trace window
    if args.trace_window > 0:
        start_time = min(r.arrival_time_seconds for r in requests)
        end_time = start_time + args.trace_window * 60
        requests = [r for r in requests if r.arrival_time_seconds <= end_time]
    
    print(f"Loaded {len(requests)} requests")
    
    # Run simulation
    config = {
        "initial_replicas": args.initial_replicas,
        "min_replicas": args.min_replicas,
        "max_replicas": args.max_replicas,
        "startup_delay": args.startup_delay,
        "cooldown": args.cooldown,
        "decision_interval": args.decision_interval,
        "episode_duration": args.episode_duration,
        "sla_threshold": args.sla_threshold,
        "replica_capacity": 10,
        "random_seed": args.seed,
        "fixed_capacity": args.fixed_capacity,
        "threshold_upper": 0.75,
        "threshold_lower": 0.25,
        "scaling_in_duration": 60,
        "queue_based_target": 0.5,
        "queue_scale_up": 1.0,
        "queue_scale_down": 0.2,
        "model_path": args.model_path,
    }
    
    policy = get_policy(args.policy, config)
    
    engine = SimulatorEngine(
        requests=requests,
        initial_replicas=args.initial_replicas,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        startup_delay=args.startup_delay,
        cooldown=args.cooldown,
        decision_interval=args.decision_interval,
        episode_duration=args.episode_duration,
        sla_threshold=args.sla_threshold,
        replica_capacity=10,
        seed=args.seed,
    )
    
    events = []
    step_count = 0
    max_steps = int(args.episode_duration / args.decision_interval)
    
    while engine.events and engine.current_time < args.episode_duration:
        result = engine.step()
        step_count += 1
        
        if result and result.get("decision_point"):
            obs = result["observation"]
            obs["time"] = engine.current_time
            action = policy.act(obs, engine)
            engine.take_action(action)
            events.append({"time": engine.current_time, "action": action, "observation": obs})
        
        if step_count % 100 == 0:
            print(f"  Progress: {step_count}/{max_steps} steps ({engine.current_time:.1f}s)")
    
    metrics = engine.get_metrics()
    timeline = engine.get_timeline()
    
    print("\n=== Results ===")
    print(f"Total Cost: ${metrics['total_cost']:.2f}")
    print(f"Avg Active Replicas: {metrics['avg_active_replicas']:.2f}")
    print(f"p50 Latency: {metrics['p50_latency_ms']:.1f}ms")
    print(f"p95 Latency: {metrics['p95_latency_ms']:.1f}ms")
    print(f"p99 Latency: {metrics['p99_latency_ms']:.1f}ms")
    print(f"SLA Violation Rate: {metrics['sla_violation_rate']*100:.1f}%")
    print(f"Goodput: {metrics['goodput']:.2f} req/s")
    print(f"Throughput: {metrics['throughput']:.2f} req/s")
    print(f"Max Queue Depth: {metrics['max_queue_depth']}")
    print(f"Avg Queue Depth: {metrics['avg_queue_depth']:.1f}")
    print(f"Scaling Events: {metrics['scaling_events']}")
    print(f"Unfinished Requests: {metrics['unfinished_requests']}")
    
    # Save results
    if args.output:
        output_data = {
            "config": config,
            "metrics": metrics,
            "timeline": timeline,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def train_ppo(args):
    """Train a PPO model."""
    print("Starting PPO training...")
    print(f"Timesteps: {args.timesteps}")
    print(f"Short run: {args.short}")
    
    # Generate training data
    if args.short:
        timesteps = 5000
    else:
        timesteps = args.timesteps
    
    # Create synthetic requests for training
    requests = []
    for wt in ["steady", "periodic", "flash_crowd"]:
        requests.extend(SYNTHETIC_WORKLOADS[wt](num_requests=2000))
    
    env_config = {
        "initial_replicas": 2,
        "min_replicas": 1,
        "max_replicas": 8,
        "startup_delay": 10.0,
        "cooldown": 30.0,
        "decision_interval": 5.0,
        "episode_duration": 3600.0,
        "sla_threshold": 30.0,
        "replica_capacity": 10,
        "random_seed": 42,
    }
    
    reward_config = {
        "alpha": 0.1,
        "beta": 1.0,
        "gamma": 0.01,
        "delta": 0.01,
    }
    
    # Create environment
    env = InferScaleEnv(requests=requests, config=env_config, reward_config=reward_config)
    
    # Validate environment
    from stable_baselines3.common.env_checker import check_env
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check warning: {e}")
    
    # Train PPO
    from stable_baselines3 import PPO
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        policy_kwargs={"net_arch": [args.hidden_units, args.hidden_units]},
    )
    
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    # Save model
    model_path = f"{settings.models_dir}/ppo_{'short' if args.short else 'full'}.zip"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "timesteps": timesteps,
        "config": env_config,
        "reward_config": reward_config,
        "ppo_config": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "hidden_units": args.hidden_units,
        },
    }
    meta_path = model_path.replace(".zip", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def evaluate_model(args):
    """Evaluate a trained model."""
    print(f"Evaluating model: {args.model_path}")
    
    # Load model
    from stable_baselines3 import PPO
    model = PPO.load(args.model_path)
    
    # Generate test requests
    requests = []
    for wt in ["steady", "periodic", "flash_crowd"]:
        requests.extend(SYNTHETIC_WORKLOADS[wt](num_requests=1000))
    
    env_config = {
        "initial_replicas": 2,
        "min_replicas": 1,
        "max_replicas": 8,
        "startup_delay": 10.0,
        "cooldown": 30.0,
        "decision_interval": 5.0,
        "episode_duration": 3600.0,
        "sla_threshold": 30.0,
        "replica_capacity": 10,
        "random_seed": args.seed,
    }
    
    reward_config = {"alpha": 0.1, "beta": 1.0, "gamma": 0.01, "delta": 0.01}
    env = InferScaleEnv(requests=requests, config=env_config, reward_config=reward_config)
    
    episode_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1}: reward={total_reward:.3f}")
    
    print(f"\nAverage reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    
    # Also run simulation with the policy for serving metrics
    from inferscale.policies.ppo import PPOPolicy
    policy = PPOPolicy({"model_path": args.model_path})
    
    engine = SimulatorEngine(
        requests=requests,
        initial_replicas=2,
        min_replicas=1,
        max_replicas=8,
        startup_delay=10.0,
        cooldown=30.0,
        decision_interval=5.0,
        episode_duration=3600.0,
        sla_threshold=30.0,
        replica_capacity=10,
        seed=args.seed,
    )
    
    while engine.events and engine.current_time < 3600.0:
        result = engine.step()
        if result and result.get("decision_point"):
            obs = result["observation"]
            obs["time"] = engine.current_time
            action = policy.act(obs, engine)
            engine.take_action(action)
    
    metrics = engine.get_metrics()
    print("\n=== Serving Metrics ===")
    print(f"Total Cost: ${metrics['total_cost']:.2f}")
    print(f"Avg Active Replicas: {metrics['avg_active_replicas']:.2f}")
    print(f"p50 Latency: {metrics['p50_latency_ms']:.1f}ms")
    print(f"p95 Latency: {metrics['p95_latency_ms']:.1f}ms")
    print(f"p99 Latency: {metrics['p99_latency_ms']:.1f}ms")
    print(f"SLA Violation Rate: {metrics['sla_violation_rate']*100:.1f}%")
    print(f"Goodput: {metrics['goodput']:.2f} req/s")
    print(f"Throughput: {metrics['throughput']:.2f} req/s")


def benchmark(args):
    """Run benchmark comparing all policies."""
    print("Running benchmark across all policies...")
    
    # Generate test workloads
    workloads = {
        "steady": SYNTHETIC_WORKLOADS["steady"](num_requests=2000),
        "periodic": SYNTHETIC_WORKLOADS["periodic"](num_requests=2000),
        "flash_crowd": SYNTHETIC_WORKLOADS["flash_crowd"](num_requests=2000),
    }
    
    policies = ["fixed", "threshold", "queue"]
    if args.include_ppo and args.ppo_model:
        policies.append("ppo")
    
    results = {}
    
    for wl_name, requests in workloads.items():
        print(f"\n--- Workload: {wl_name} ({len(requests)} requests) ---")
        results[wl_name] = {}
        
        for policy_name in policies:
            print(f"  Running {policy_name}...", end=" ", flush=True)
            
            config = {
                "initial_replicas": 2,
                "min_replicas": 1,
                "max_replicas": 8,
                "startup_delay": 10.0,
                "cooldown": 30.0,
                "decision_interval": 5.0,
                "episode_duration": 3600.0,
                "sla_threshold": 30.0,
                "replica_capacity": 10,
                "random_seed": 42,
                "fixed_capacity": 4,
                "threshold_upper": 0.75,
                "threshold_lower": 0.25,
                "scaling_in_duration": 60,
                "queue_based_target": 0.5,
                "queue_scale_up": 1.0,
                "queue_scale_down": 0.2,
                "model_path": args.ppo_model,
            }
            
            policy = get_policy(policy_name, config)
            
            engine = SimulatorEngine(
                requests=requests,
                initial_replicas=2,
                min_replicas=1,
                max_replicas=8,
                startup_delay=10.0,
                cooldown=30.0,
                decision_interval=5.0,
                episode_duration=3600.0,
                sla_threshold=30.0,
                replica_capacity=10,
                seed=42,
            )
            
            while engine.events and engine.current_time < 3600.0:
                result = engine.step()
                if result and result.get("decision_point"):
                    obs = result["observation"]
                    obs["time"] = engine.current_time
                    action = policy.act(obs, engine)
                    engine.take_action(action)
            
            metrics = engine.get_metrics()
            results[wl_name][policy_name] = metrics
            
            print(f"Cost=${metrics['total_cost']:.2f}, p95={metrics['p95_latency_ms']:.1f}ms, SLA={metrics['sla_violation_rate']*100:.1f}%")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmark results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="InferScale CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate data
    gen_parser = subparsers.add_parser("generate-data", help="Generate synthetic workload")
    gen_parser.add_argument("--workload-type", choices=["steady", "periodic", "flash_crowd", "mixed"], default="steady")
    gen_parser.add_argument("--num-requests", type=int, default=10000)
    gen_parser.add_argument("--output", default="data/synthetic.csv")
    
    # Run simulation
    sim_parser = subparsers.add_parser("simulate", help="Run a simulation")
    sim_parser.add_argument("--policy", choices=["fixed", "threshold", "queue", "ppo"], default="threshold")
    sim_parser.add_argument("--dataset", default="synthetic")
    sim_parser.add_argument("--workload-type", choices=["steady", "periodic", "flash_crowd", "mixed"], default="steady")
    sim_parser.add_argument("--trace-window", type=float, default=30)
    sim_parser.add_argument("--initial-replicas", type=int, default=2)
    sim_parser.add_argument("--min-replicas", type=int, default=1)
    sim_parser.add_argument("--max-replicas", type=int, default=8)
    sim_parser.add_argument("--startup-delay", type=float, default=10)
    sim_parser.add_argument("--cooldown", type=float, default=30)
    sim_parser.add_argument("--decision-interval", type=float, default=5)
    sim_parser.add_argument("--episode-duration", type=float, default=3600)
    sim_parser.add_argument("--sla-threshold", type=float, default=30)
    sim_parser.add_argument("--seed", type=int, default=42)
    sim_parser.add_argument("--fixed-capacity", type=int, default=4)
    sim_parser.add_argument("--model-path", help="Path to PPO model (for ppo policy)")
    sim_parser.add_argument("--output", help="Output file for results")
    
    # Train PPO
    train_parser = subparsers.add_parser("train", help="Train PPO model")
    train_parser.add_argument("--timesteps", type=int, default=200000)
    train_parser.add_argument("--short", action="store_true", help="Short smoke test (5000 timesteps)")
    train_parser.add_argument("--n-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--hidden-units", type=int, default=128)
    
    # Evaluate model
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model-path", required=True, help="Path to trained model")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--seed", type=int, default=42)
    
    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run policy comparison benchmark")
    bench_parser.add_argument("--output", default="results/benchmark.json")
    bench_parser.add_argument("--include-ppo", action="store_true")
    bench_parser.add_argument("--ppo-model", help="Path to PPO model for benchmark")
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        generate_data(args)
    elif args.command == "simulate":
        run_sim(args)
    elif args.command == "train":
        train_ppo(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()