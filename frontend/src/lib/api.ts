import axios, { AxiosInstance, AxiosError } from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    })

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        console.error('API Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  // Health
  async health(): Promise<{ status: string; timestamp: string }> {
    const res = await this.client.get('/health')
    return res.data
  }

  // Datasets
  async listDatasets(): Promise<Dataset[]> {
    const res = await this.client.get('/datasets')
    return res.data
  }

  async getDatasetStats(id: string): Promise<DatasetStats> {
    const res = await this.client.get(`/datasets/${id}/stats`)
    return res.data
  }

  async validateDataset(file: File): Promise<ValidationResult> {
    const formData = new FormData()
    formData.append('file', file)
    const res = await this.client.post('/datasets/validate', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return res.data
  }

  // Experiments
  async createExperiment(config: ExperimentConfig): Promise<ExperimentJob> {
    const res = await this.client.post('/experiments', config)
    return res.data
  }

  async listExperiments(status?: string): Promise<ExperimentJob[]> {
    const params = status ? { status } : {}
    const res = await this.client.get('/experiments', { params })
    return res.data
  }

  async getExperiment(id: string): Promise<ExperimentJob> {
    const res = await this.client.get(`/experiments/${id}`)
    return res.data
  }

  async cancelExperiment(id: string): Promise<void> {
    await this.client.post(`/experiments/${id}/cancel`)
  }

  async getExperimentMetrics(id: string): Promise<ExperimentMetrics> {
    const res = await this.client.get(`/experiments/${id}/metrics`)
    return res.data
  }

  async getExperimentTimeline(id: string): Promise<TimelineEvent[]> {
    const res = await this.client.get(`/experiments/${id}/timeline`)
    return res.data
  }

  async exportExperiment(id: string, format: 'csv' | 'json' = 'json'): Promise<Blob> {
    const res = await this.client.get(`/experiments/${id}/export`, {
      params: { format },
      responseType: 'blob',
    })
    return res.data
  }

  // Training
  async startTraining(config: TrainingConfig): Promise<TrainingJob> {
    const res = await this.client.post('/training', config)
    return res.data
  }

  async getTraining(id: string): Promise<TrainingJob> {
    const res = await this.client.get(`/training/${id}`)
    return res.data
  }

  async listModels(): Promise<ModelInfo[]> {
    const res = await this.client.get('/models')
    return res.data
  }

  // Evaluations
  async startEvaluation(config: EvaluationConfig): Promise<EvaluationJob> {
    const res = await this.client.post('/evaluations', config)
    return res.data
  }

  // Reports
  async getReport(id: string): Promise<Report> {
    const res = await this.client.get(`/reports/${id}`)
    return res.data
  }

  // SSE for live updates
  createEventSource(endpoint: string): EventSource {
    return new EventSource(`${API_BASE}${endpoint}`)
  }
}

export const api = new ApiClient()

// Types
export interface Dataset {
  id: string
  name: string
  source: string
  record_count: number
  duration_seconds: number
  created_at: string
  hash: string
}

export interface DatasetStats {
  id: string
  record_count: number
  duration_seconds: number
  prompt_tokens: TokenStats
  output_tokens: TokenStats
  arrival_rate: TimeSeriesPoint[]
  burst_detection: BurstInfo[]
  missing_values: number
  invalid_rows: number
  schema: Record<string, string>
}

export interface TokenStats {
  min: number
  max: number
  mean: number
  median: number
  std: number
  p50: number
  p95: number
  p99: number
}

export interface TimeSeriesPoint {
  timestamp: number
  value: number
}

export interface BurstInfo {
  start_time: number
  end_time: number
  peak_rate: number
  avg_rate: number
  duration: number
}

export interface ValidationResult {
  valid: boolean
  record_count: number
  errors: string[]
  warnings: string[]
  preview: Record<string, unknown>[]
}

export interface ExperimentConfig {
  workload_type: string
  dataset_id: string
  trace_window_minutes: number
  policy: string
  initial_replicas: number
  min_replicas: number
  max_replicas: number
  replica_startup_delay: number
  scaling_cooldown: number
  decision_interval: number
  episode_duration: number
  sla_threshold: number
  random_seed: number
  reward_alpha: number
  reward_beta: number
  reward_gamma: number
  reward_delta: number
  ppo_timesteps: number
  ppo_alpha: number
  ppo_beta: number
  ppo_gamma: number
  short_run: boolean
}

export interface ExperimentJob {
  id: string
  config: ExperimentConfig
  state: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed'
  progress: number
  started_at: string | null
  completed_at: string | null
  error: string | null
  results: ExperimentMetrics | null
  timeline: TimelineEvent[]
}

export interface ExperimentMetrics {
  experiment_id: string
  total_cost: number
  avg_active_replicas: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  sla_violation_rate: number
  goodput: number
  throughput: number
  max_queue_depth: number
  avg_queue_depth: number
  scaling_events: number
  burst_recovery_time: number
  unfinished_requests: number
  episode_reward: number
  timestamp: string
}

export interface TimelineEvent {
  type: 'request' | 'action'
  // Request fields
  id?: string
  arrival?: number
  queue_entry?: number
  prefill_start?: number
  prefill_end?: number
  decode_start?: number
  decode_end?: number
  replica?: number
  sla_violated?: boolean
  // Action fields
  time?: number
  action?: string
  replicas?: number
}

export interface TrainingConfig {
  timesteps: number
  n_steps: number
  batch_size: number
  n_epochs: number
  learning_rate: number
  gamma: number
  hidden_units: number
  reward_alpha: number
  reward_beta: number
  reward_gamma: number
  reward_delta: number
  dataset_id: string
  workload_type: string
  short_run: boolean
}

export interface TrainingJob {
  id: string
  config: TrainingConfig
  state: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed'
  progress: number
  current_step: number
  total_steps: number
  episode_rewards: number[]
  eval_rewards: number[]
  started_at: string | null
  completed_at: string | null
  error: string | null
  model_path: string | null
}

export interface ModelInfo {
  id: string
  name: string
  path: string
  config: TrainingConfig
  created_at: string
  metrics: TrainingMetrics | null
}

export interface TrainingMetrics {
  final_reward: number
  avg_eval_reward: number
  training_time_seconds: number
}

export interface EvaluationConfig {
  model_id: string
  dataset_id: string
  workload_type: string
  episodes: number
}

export interface EvaluationJob {
  id: string
  config: EvaluationConfig
  state: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed'
  progress: number
  results: ExperimentMetrics[] | null
  started_at: string | null
  completed_at: string | null
  error: string | null
}

export interface Report {
  id: string
  experiment_ids: string[]
  markdown: string
  created_at: string
}