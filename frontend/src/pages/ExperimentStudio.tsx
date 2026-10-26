import { useState } from 'react'
import { useDatasets } from '../hooks/useData'
import { api, ExperimentConfig } from '../lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Select } from '../components/ui/select'
import { useToast } from '../components/ui/toast'
import { Save, Play, Loader2 } from 'lucide-react'

const WORKLOAD_TYPES = [
  { value: 'steady', label: 'Steady' },
  { value: 'periodic', label: 'Periodic' },
  { value: 'flash_crowd', label: 'Flash Crowd' },
  { value: 'mixed', label: 'Mixed' },
]

const POLICIES = [
  { value: 'fixed', label: 'Fixed Capacity' },
  { value: 'threshold', label: 'Threshold Autoscaling' },
  { value: 'queue', label: 'Queue-Based Autoscaling' },
  { value: 'ppo', label: 'PPO (RL)' },
]

const DATASETS = [
  { value: 'synthetic', label: 'Synthetic' },
]

export function ExperimentStudio() {
  const { datasets } = useDatasets()
  const { toast } = useToast()
  const [submitting, setSubmitting] = useState(false)
  const [shortRun, setShortRun] = useState(true)

  const [config, setConfig] = useState<ExperimentConfig>({
    workload_type: 'steady',
    dataset_id: 'synthetic',
    trace_window_minutes: 30,
    policy: 'threshold',
    initial_replicas: 2,
    min_replicas: 1,
    max_replicas: 8,
    replica_startup_delay: 10,
    scaling_cooldown: 30,
    decision_interval: 5,
    episode_duration: 3600,
    sla_threshold: 30,
    slo_target: 0.99,
    cost_per_replica_second: 0.01,
    random_seed: 42,
    reward_alpha: 0.1,
    reward_beta: 1.0,
    reward_gamma: 0.01,
    reward_delta: 0.01,
    ppo_timesteps: 100000,
    ppo_alpha: 0.1,
    ppo_beta: 0.01,
    ppo_gamma: 0.99,
    short_run: true,
  })

  const handleChange = (field: keyof ExperimentConfig, value: string | number | boolean) => {
    setConfig((prev) => ({ ...prev, [field]: value, short_run: shortRun }))
  }

  const handleSubmit = async (e: React.FormEvent, run: boolean) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      const finalConfig = { ...config, short_run: shortRun }
      if (run) {
        const res = await api.createExperiment(finalConfig)
        toast(`Experiment started: ${res.id}`, 'success')
        window.location.href = `/simulation/${res.id}`
      } else {
        toast('Configuration saved', 'success')
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to create experiment', 'destructive')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Experiment Studio</h1>
          <p className="text-muted-foreground">Configure and launch autoscaling experiments</p>
        </div>
        <div className="flex items-center gap-2">
          <Label className="cursor-pointer flex items-center gap-2">
            <input
              type="checkbox"
              checked={shortRun}
              onChange={(e) => setShortRun(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
            />
            Short Run Mode (5 min episode)
          </Label>
        </div>
      </div>

      <form onSubmit={(e) => handleSubmit(e, true)} className="space-y-6">
        {/* Workload Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Workload Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="workload_type">Workload Type</Label>
                <Select
                  id="workload_type"
                  value={config.workload_type}
                  onValueChange={(v) => handleChange('workload_type', v)}
                >
                  {WORKLOAD_TYPES.map((w) => (
                    <option key={w.value} value={w.value}>{w.label}</option>
                  ))}
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="dataset_id">Dataset</Label>
                <Select
                  id="dataset_id"
                  value={config.dataset_id}
                  onValueChange={(v) => handleChange('dataset_id', v)}
                >
                  {DATASETS.map((d) => (
                    <option key={d.value} value={d.value}>{d.label}</option>
                  ))}
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>{d.name} ({d.source})</option>
                  ))}
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="trace_window_minutes">Trace Window (minutes)</Label>
                <Input
                  id="trace_window_minutes"
                  type="number"
                  value={config.trace_window_minutes}
                  onChange={(e) => handleChange('trace_window_minutes', parseFloat(e.target.value))}
                  min="1"
                  max="1440"
                  step="1"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="policy">Scaling Policy</Label>
                <Select
                  id="policy"
                  value={config.policy}
                  onValueChange={(v) => handleChange('policy', v)}
                >
                  {POLICIES.map((p) => (
                    <option key={p.value} value={p.value}>{p.label}</option>
                  ))}
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Replica Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Replica Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="initial_replicas">Initial Replicas</Label>
                <Input
                  id="initial_replicas"
                  type="number"
                  value={config.initial_replicas}
                  onChange={(e) => handleChange('initial_replicas', parseInt(e.target.value))}
                  min="1"
                  max="8"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="min_replicas">Min Replicas</Label>
                <Input
                  id="min_replicas"
                  type="number"
                  value={config.min_replicas}
                  onChange={(e) => handleChange('min_replicas', parseInt(e.target.value))}
                  min="1"
                  max="8"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max_replicas">Max Replicas</Label>
                <Input
                  id="max_replicas"
                  type="number"
                  value={config.max_replicas}
                  onChange={(e) => handleChange('max_replicas', parseInt(e.target.value))}
                  min="1"
                  max="16"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="replica_startup_delay">Startup Delay (seconds)</Label>
                <Input
                  id="replica_startup_delay"
                  type="number"
                  value={config.replica_startup_delay}
                  onChange={(e) => handleChange('replica_startup_delay', parseFloat(e.target.value))}
                  min="0"
                  max="300"
                  step="1"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="scaling_cooldown">Scaling Cooldown (seconds)</Label>
                <Input
                  id="scaling_cooldown"
                  type="number"
                  value={config.scaling_cooldown}
                  onChange={(e) => handleChange('scaling_cooldown', parseFloat(e.target.value))}
                  min="0"
                  max="600"
                  step="1"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="replica_capacity">Replica Capacity (concurrent requests)</Label>
                <Input
                  id="replica_capacity"
                  type="number"
                  value={10}
                  onChange={(e) => handleChange('replica_capacity' as keyof ExperimentConfig, parseInt(e.target.value))}
                  min="1"
                  max="100"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Simulation Parameters */}
        <Card>
          <CardHeader>
            <CardTitle>Simulation Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="decision_interval">Decision Interval (seconds)</Label>
                <Input
                  id="decision_interval"
                  type="number"
                  value={config.decision_interval}
                  onChange={(e) => handleChange('decision_interval', parseFloat(e.target.value))}
                  min="0.1"
                  max="60"
                  step="0.5"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="episode_duration">Episode Duration (seconds)</Label>
                <Input
                  id="episode_duration"
                  type="number"
                  value={config.episode_duration}
                  onChange={(e) => handleChange('episode_duration', parseFloat(e.target.value))}
                  min="60"
                  max="86400"
                  step="60"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="sla_threshold">SLA Threshold (seconds)</Label>
                <Input
                  id="sla_threshold"
                  type="number"
                  value={config.sla_threshold}
                  onChange={(e) => handleChange('sla_threshold', parseFloat(e.target.value))}
                  min="1"
                  max="300"
                  step="1"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="slo_target">SLO target (% of arrivals)</Label>
                <Input id="slo_target" type="number" min={0} max={100} step={0.1}
                  value={(config.slo_target ?? 0.99) * 100}
                  onChange={e => handleChange('slo_target', Number(e.target.value) / 100)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="cost_per_replica_second">USD per replica-second</Label>
                <Input id="cost_per_replica_second" type="number" min={0} step="any"
                  value={config.cost_per_replica_second ?? 0.01}
                  onChange={e => handleChange('cost_per_replica_second', Number(e.target.value))} />
                <p className="text-xs text-muted-foreground">Billed while starting, active (including idle), or draining.</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="random_seed">Random Seed</Label>
                <Input
                  id="random_seed"
                  type="number"
                  value={config.random_seed}
                  onChange={(e) => handleChange('random_seed', parseInt(e.target.value))}
                  min="0"
                  max="2147483647"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Reward Weights */}
        <Card>
          <CardHeader>
            <CardTitle>Reward Weights (PPO Training)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              r = -(α·C + β·V + γ·Q + δ·S) where C=cost, V=SLA violation, Q=queue pressure, S=scaling churn
            </p>
            <div className="grid gap-4 md:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="reward_alpha">α (Cost Weight)</Label>
                <Input
                  id="reward_alpha"
                  type="number"
                  value={config.reward_alpha}
                  onChange={(e) => handleChange('reward_alpha', parseFloat(e.target.value))}
                  min="0"
                  max="10"
                  step="0.01"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="reward_beta">β (SLA Violation Weight)</Label>
                <Input
                  id="reward_beta"
                  type="number"
                  value={config.reward_beta}
                  onChange={(e) => handleChange('reward_beta', parseFloat(e.target.value))}
                  min="0"
                  max="10"
                  step="0.01"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="reward_gamma">γ (Queue Pressure Weight)</Label>
                <Input
                  id="reward_gamma"
                  type="number"
                  value={config.reward_gamma}
                  onChange={(e) => handleChange('reward_gamma', parseFloat(e.target.value))}
                  min="0"
                  max="10"
                  step="0.001"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="reward_delta">δ (Scaling Churn Weight)</Label>
                <Input
                  id="reward_delta"
                  type="number"
                  value={config.reward_delta}
                  onChange={(e) => handleChange('reward_delta', parseFloat(e.target.value))}
                  min="0"
                  max="10"
                  step="0.001"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* PPO Training Config */}
        <Card>
          <CardHeader>
            <CardTitle>PPO Training Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="ppo_timesteps">Training Timesteps</Label>
                <Input
                  id="ppo_timesteps"
                  type="number"
                  value={config.ppo_timesteps}
                  onChange={(e) => handleChange('ppo_timesteps', parseInt(e.target.value))}
                  min="1000"
                  max="1000000"
                  step="1000"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ppo_alpha">PPO Reward α</Label>
                <Input
                  id="ppo_alpha"
                  type="number"
                  value={config.ppo_alpha}
                  onChange={(e) => handleChange('ppo_alpha', parseFloat(e.target.value))}
                  min="0"
                  max="10"
                  step="0.01"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ppo_beta">PPO Reward β</Label>
                <Input
                  id="ppo_beta"
                  type="number"
                  value={config.ppo_beta}
                  onChange={(e) => handleChange('ppo_beta', parseFloat(e.target.value))}
                  min="0"
                  max="10"
                  step="0.01"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ppo_gamma">PPO Gamma</Label>
                <Input
                  id="ppo_gamma"
                  type="number"
                  value={config.ppo_gamma}
                  onChange={(e) => handleChange('ppo_gamma', parseFloat(e.target.value))}
                  min="0.9"
                  max="0.999"
                  step="0.001"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Actions */}
        <div className="flex items-center gap-4 pt-4 border-t">
          <Button type="submit" disabled={submitting} className="gap-2">
            <Play className="h-4 w-4" />
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Launch Experiment'}
          </Button>
          <Button type="button" variant="outline" onClick={(e) => handleSubmit(e, false)} disabled={submitting} className="gap-2">
            <Save className="h-4 w-4" />
            Save Config
          </Button>
        </div>
      </form>
    </div>
  )
}
