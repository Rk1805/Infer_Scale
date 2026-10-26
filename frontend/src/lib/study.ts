import { api } from './api'

export type StudyProfile = 'bursty' | 'steady' | 'shifted'
export type StudyPolicy = 'fixed' | 'threshold' | 'queue' | 'ppo'
export interface StudyStats { n: number; mean: number; ci_low: number | null; ci_high: number | null }
export interface StudySummary {
  profile: StudyProfile; policy: StudyPolicy; run_count: number; slo_met_runs: number
  total_cost: StudyStats; p95_latency_ms: StudyStats; slo_attainment_lower_bound: StudyStats
}
export interface StudyMetrics {
  total_cost: number; p95_latency_ms: number; slo_attainment_lower_bound: number
  on_time_completions: number; late_completions: number; overdue_requests: number
  pending_requests: number; scaling_events: number; arrivals: number
}
export interface StudyTimeline { time: number; action: number; active: number; starting: number; queue: number }
export interface StudyReplay {
  profile: StudyProfile; seed: number; request_count: number; trace_hash: string
  checkpoint_sha256: string; source: string
  runs: Array<{ policy: StudyPolicy; metrics: StudyMetrics; timeline: StudyTimeline[]; matches_recorded: boolean }>
}
export interface StudyData {
  checkpoint: { name: string; sha256: string; verified: boolean; selected_step: number }
  manifest: {
    training_steps: number; selected_step: number; ppo_seed: number; validation_return: number
    training_seeds: number[]; validation_seeds: number[]; test_seeds: number[]
    config: { episode_duration: number; decision_interval: number; sla_threshold: number; slo_target: number; initial_replicas: number; min_replicas: number; max_replicas: number; cooldown: number; startup_delay: number; replica_capacity: number }
    policy: { ent_coef: number; learning_rate: number; n_steps: number; batch_size: number; n_epochs: number; gamma: number }
    rates: Record<StudyProfile, number[]>
  }
  profiles: StudyProfile[]; seeds: number[]; summaries: StudySummary[]
  runs: Array<{ profile: StudyProfile; policy: StudyPolicy; seed: number; trace_hash: string; metrics: StudyMetrics }>
  learning: Array<{ timesteps: number; training_return?: number; validation_return?: number }>
  limitations: string[]
}

export const studyApi = {
  get: () => api.getStudy(),
  replay: (profile: StudyProfile, seed: number) => api.replayStudy(profile, seed),
}
