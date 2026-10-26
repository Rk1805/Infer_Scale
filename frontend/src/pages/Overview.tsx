import { useHealth, useExperiments, useModels, useDatasets } from '../hooks/useData'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Separator } from '../components/ui/separator'
import { Progress } from '../components/ui/progress'
import { useToast } from '../components/ui/toast'
import { api } from '../lib/api'
import {
CheckCircle,
AlertCircle,
Loader2,Brain,
FlaskConical,
GitCompare,TrendingUp,Database
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { formatPercent } from '../lib/utils'

export function Overview() {
  const { health, loading: healthLoading } = useHealth()
  const { experiments } = useExperiments()
  const { models } = useModels()
  const { datasets } = useDatasets()
  const { toast } = useToast()

  const latestExperiment = experiments[0]
  const latestModel = models[0]

  const handleQuickAction = async (action: string) => {
    try {
      if (action === 'train') {
        const res = await api.startTraining({
          timesteps: 5000,
          n_steps: 512,
          batch_size: 64,
          n_epochs: 10,
          learning_rate: 3e-4,
          gamma: 0.99,
          hidden_units: 128,
          reward_alpha: 0.1,
          reward_beta: 1.0,
          reward_gamma: 0.01,
          reward_delta: 0.01,
          dataset_id: 'synthetic',
          workload_type: 'mixed',
          short_run: true,
        })
        toast(`Training started: ${res.id}`, 'success')
        window.location.href = `/training-lab`
      } else if (action === 'simulate') {
        const res = await api.createExperiment({
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
        toast(`Simulation started: ${res.id}`, 'success')
        window.location.href = `/simulation/${res.id}`
      } else if (action === 'compare') {
        window.location.href = '/policy-comparison'
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Action failed', 'destructive')
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default"><CheckCircle className="mr-1 h-3 w-3" /> Completed</Badge>
      case 'running':
        return <Badge variant="secondary"><Loader2 className="mr-1 h-3 w-3 animate-spin" /> Running</Badge>
      case 'failed':
        return <Badge variant="destructive"><AlertCircle className="mr-1 h-3 w-3" /> Failed</Badge>
      case 'cancelled':
        return <Badge variant="outline">Cancelled</Badge>
      default:
        return <Badge variant="outline">Queued</Badge>
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Overview</h1>
          <p className="text-muted-foreground">InferScale dashboard - LLM inference autoscaling with RL</p>
        </div>
        <div className="flex items-center gap-2">
          {healthLoading ? (
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          ) : health?.status === 'ok' ? (
            <Badge variant="default"><CheckCircle className="mr-1 h-3 w-3" /> API Healthy</Badge>
          ) : (
            <Badge variant="destructive"><AlertCircle className="mr-1 h-3 w-3" /> API Unhealthy</Badge>
          )}
        </div>
      </div>

      <Separator />

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasets.length}</div>
            <p className="text-xs text-muted-foreground">Available datasets</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Experiments</CardTitle>
            <FlaskConical className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{experiments.length}</div>
            <p className="text-xs text-muted-foreground">Total experiments run</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Models</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{models.length}</div>
            <p className="text-xs text-muted-foreground">Trained PPO models</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Latest Experiment</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {latestExperiment ? (
                <>
                  {getStatusBadge(latestExperiment.state)}
                  <span className="ml-2 text-xs text-muted-foreground">
                    {latestExperiment.started_at && formatDistanceToNow(new Date(latestExperiment.started_at), { addSuffix: true })}
                  </span>
                </>
              ) : (
                <span className="text-muted-foreground">No experiments yet</span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Latest Experiment Details */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Latest Experiment Status</CardTitle>
          </CardHeader>
          <CardContent>
            {latestExperiment ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  {getStatusBadge(latestExperiment.state)}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Policy</span>
                  <Badge variant="outline">{latestExperiment.config.policy}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Workload</span>
                  <Badge variant="outline">{latestExperiment.config.workload_type}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Progress</span>
                  <div className="w-48">
                    <Progress value={latestExperiment.progress * 100} />
                  </div>
                </div>
                {latestExperiment.results && (
                  <Separator />
                )}
                {latestExperiment.results && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Total Cost</span>
                      <div className="font-medium">${latestExperiment.results.total_cost.toFixed(2)}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Avg Replicas</span>
                      <div className="font-medium">{latestExperiment.results.avg_active_replicas.toFixed(1)}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">p95 Latency</span>
                      <div className="font-medium">{latestExperiment.results.p95_latency_ms.toFixed(1)}ms</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">SLA Violations</span>
                      <div className="font-medium">{formatPercent(latestExperiment.results.sla_violation_rate)}</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-muted-foreground">No experiments run yet. Create one in Experiment Studio.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Latest PPO Model</CardTitle>
          </CardHeader>
          <CardContent>
            {latestModel ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <Badge variant="outline">Registered</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Timesteps</span>
                  <Badge variant="outline">{latestModel.config.timesteps.toLocaleString()}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Dataset</span>
                  <Badge variant="outline">{latestModel.config.dataset_id}</Badge>
                </div>
                {latestModel.metrics && (
                  <Separator />
                )}
                {latestModel.metrics && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Final Reward</span>
                      <div className="font-medium">{latestModel.metrics.final_reward.toFixed(3)}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Avg Eval Reward</span>
                      <div className="font-medium">{latestModel.metrics.avg_eval_reward.toFixed(3)}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Training Time</span>
                      <div className="font-medium">{latestModel.metrics.training_time_seconds.toFixed(0)}s</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-muted-foreground">No models trained yet. Start training in Training Lab.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Button onClick={() => handleQuickAction('train')} className="gap-2">
              <Brain className="h-4 w-4" />
              Train PPO (Smoke Test)
            </Button>
            <Button onClick={() => handleQuickAction('simulate')} variant="secondary" className="gap-2">
              <FlaskConical className="h-4 w-4" />
              Run Simulation
            </Button>
            <Button onClick={() => handleQuickAction('compare')} variant="outline" className="gap-2">
              <GitCompare className="h-4 w-4" />
              Compare Policies
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Recent Experiments */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Recent Experiments</CardTitle>
          <Button variant="ghost" size="sm" onClick={() => window.location.href = '/experiment-history'}>
            View All
          </Button>
        </CardHeader>
        <CardContent>
          {experiments.length > 0 ? (
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {experiments.slice(0, 10).map((exp) => (
                <div key={exp.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="p-2 rounded-lg bg-muted">
                      <FlaskConical className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">{exp.id}</p>
                      <p className="text-xs text-muted-foreground">
                        {exp.config.policy} • {exp.config.workload_type} • {exp.config.dataset_id}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground">
                      {exp.started_at && formatDistanceToNow(new Date(exp.started_at), { addSuffix: true })}
                    </span>
                    {getStatusBadge(exp.state)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8">No experiments yet</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
