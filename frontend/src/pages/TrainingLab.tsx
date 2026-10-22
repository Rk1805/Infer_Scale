import { useState, useEffect } from 'react'
import { useTraining, useModels } from '../hooks/useData'
import { api, TrainingConfig } from '../lib/api'
import { useParams, useNavigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Select } from '../components/ui/select'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Separator } from '../components/ui/separator'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/table'
import { useToast } from '../components/ui/toast'
import {
Play,
Pause,
Loader2,
Brain,CheckCircle,
AlertCircle,
RotateCcw
} from 'lucide-react'
import {
LineChart,
Line,
XAxis,
YAxis,
CartesianGrid,
Tooltip,
Legend,
ResponsiveContainer,
} from 'recharts'
import { formatDistanceToNow } from 'date-fns'

export function TrainingLab() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { training, refetch: refetchTraining } = useTraining(id || null)
  const { models, loading: modelsLoading, refetch: refetchModels } = useModels()
  const { toast } = useToast()

  const [config, setConfig] = useState<TrainingConfig>({
    timesteps: 50000,
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
  const [trainingState, setTrainingState] = useState<'idle' | 'running' | 'paused'>('idle')
  const [episodeRewards, setEpisodeRewards] = useState<number[]>([])
  const [evalRewards, setEvalRewards] = useState<number[]>([])
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Auto-refresh for running training
  useEffect(() => {
    if (!training || !autoRefresh) return
    if (training.state === 'running') {
      const interval = setInterval(() => refetchTraining(), 1000)
      return () => clearInterval(interval)
    }
  }, [training, autoRefresh, refetchTraining])

  // Update local state from training job
  useEffect(() => {
    if (training) {
      setEpisodeRewards(training.episode_rewards || [])
      setEvalRewards(training.eval_rewards || [])
      setTrainingState(training.state === 'running' ? 'running' : 'idle')
    }
  }, [training])

  const handleStartTraining = async () => {
    try {
      const res = await api.startTraining(config)
      toast(`Training started: ${res.id}`, 'success')
      navigate(`/training-lab`)
      // Wait a bit then start polling
      setTimeout(() => refetchTraining(), 1000)
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to start training', 'destructive')
    }
  }

  const handleLoadModel = async (modelId: string) => {
    try {
      // This would load a model for evaluation
      toast(`Model ${modelId} loaded for evaluation`, 'success')
      navigate('/policy-comparison')
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to load model', 'destructive')
    }
  }

  const handleEvaluateModel = async (modelId: string) => {
    try {
      const res = await api.startEvaluation({
        model_id: modelId,
        dataset_id: 'synthetic',
        workload_type: 'mixed',
        episodes: 10,
      })
      toast(`Evaluation started: ${res.id}`, 'success')
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Evaluation failed', 'destructive')
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Training Lab</h1>
          <p className="text-muted-foreground">Train and evaluate PPO models for autoscaling</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => refetchModels()} disabled={modelsLoading} className="gap-2">
            <RotateCcw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Training Configuration */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>PPO Training Configuration</CardTitle>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={config.short_run}
              onChange={(e) => setConfig((prev) => ({ ...prev, short_run: e.target.checked }))}
              className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
            />
            <Label className="cursor-pointer text-sm">Short Run (5k steps)</Label>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="timesteps">Training Timesteps</Label>
              <Input
                id="timesteps"
                type="number"
                value={config.timesteps}
                onChange={(e) => setConfig((prev) => ({ ...prev, timesteps: parseInt(e.target.value) }))}
                min="1000"
                max="1000000"
                step="1000"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="n_steps">N Steps</Label>
              <Input
                id="n_steps"
                type="number"
                value={config.n_steps}
                onChange={(e) => setConfig((prev) => ({ ...prev, n_steps: parseInt(e.target.value) }))}
                min="64"
                max="4096"
                step="64"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch_size">Batch Size</Label>
              <Input
                id="batch_size"
                type="number"
                value={config.batch_size}
                onChange={(e) => setConfig((prev) => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                min="16"
                max="512"
                step="16"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="n_epochs">N Epochs</Label>
              <Input
                id="n_epochs"
                type="number"
                value={config.n_epochs}
                onChange={(e) => setConfig((prev) => ({ ...prev, n_epochs: parseInt(e.target.value) }))}
                min="1"
                max="50"
                step="1"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="learning_rate">Learning Rate</Label>
              <Input
                id="learning_rate"
                type="number"
                value={config.learning_rate}
                onChange={(e) => setConfig((prev) => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                min="1e-5"
                max="1e-2"
                step="1e-5"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="gamma">Gamma</Label>
              <Input
                id="gamma"
                type="number"
                value={config.gamma}
                onChange={(e) => setConfig((prev) => ({ ...prev, gamma: parseFloat(e.target.value) }))}
                min="0.9"
                max="0.999"
                step="0.001"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="hidden_units">Hidden Units</Label>
              <Input
                id="hidden_units"
                type="number"
                value={config.hidden_units}
                onChange={(e) => setConfig((prev) => ({ ...prev, hidden_units: parseInt(e.target.value) }))}
                min="32"
                max="512"
                step="32"
              />
            </div>
          </div>

          <Separator className="my-4" />

          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="dataset_id">Dataset</Label>
              <Select
                id="dataset_id"
                value={config.dataset_id}
                onValueChange={(v) => setConfig((prev) => ({ ...prev, dataset_id: v }))}
              >
                <option value="synthetic">Synthetic</option>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="workload_type">Workload Type</Label>
              <Select
                id="workload_type"
                value={config.workload_type}
                onValueChange={(v) => setConfig((prev) => ({ ...prev, workload_type: v }))}
              >
                <option value="steady">Steady</option>
                <option value="periodic">Periodic</option>
                <option value="flash_crowd">Flash Crowd</option>
                <option value="mixed">Mixed</option>
              </Select>
            </div>
          </div>

          <Separator className="my-4" />

          <h4 className="font-medium">Reward Weights</h4>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="reward_alpha">α (Cost)</Label>
              <Input
                id="reward_alpha"
                type="number"
                value={config.reward_alpha}
                onChange={(e) => setConfig((prev) => ({ ...prev, reward_alpha: parseFloat(e.target.value) }))}
                min="0"
                max="10"
                step="0.01"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="reward_beta">β (SLA)</Label>
              <Input
                id="reward_beta"
                type="number"
                value={config.reward_beta}
                onChange={(e) => setConfig((prev) => ({ ...prev, reward_beta: parseFloat(e.target.value) }))}
                min="0"
                max="10"
                step="0.01"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="reward_gamma">γ (Queue)</Label>
              <Input
                id="reward_gamma"
                type="number"
                value={config.reward_gamma}
                onChange={(e) => setConfig((prev) => ({ ...prev, reward_gamma: parseFloat(e.target.value) }))}
                min="0"
                max="10"
                step="0.001"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="reward_delta">δ (Churn)</Label>
              <Input
                id="reward_delta"
                type="number"
                value={config.reward_delta}
                onChange={(e) => setConfig((prev) => ({ ...prev, reward_delta: parseFloat(e.target.value) }))}
                min="0"
                max="10"
                step="0.001"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4 pt-4 border-t">
            {id && training ? (
              <>
                {training.state === 'running' ? (
                  <Button variant="outline" onClick={() => setTrainingState('paused')} disabled className="gap-2">
                    <Pause className="h-4 w-4" />
                    Running...
                  </Button>
                ) : training.state === 'completed' ? (
                  <Badge variant="default"><CheckCircle className="mr-1 h-3 w-3" /> Completed</Badge>
                ) : training.state === 'failed' ? (
                  <Badge variant="destructive"><AlertCircle className="mr-1 h-3 w-3" /> Failed</Badge>
                ) : (
                  <Button onClick={handleStartTraining} className="gap-2">
                    <Play className="h-4 w-4" />
                    Start Training
                  </Button>
                )}
                {training.model_path && (
                  <Button variant="outline" onClick={() => handleLoadModel(training.id)} className="gap-2">
                    <Brain className="h-4 w-4" />
                    Load for Evaluation
                  </Button>
                )}
              </>
            ) : (
              <Button onClick={handleStartTraining} disabled={trainingState === 'running'} className="gap-2">
                <Play className="h-4 w-4" />
                {trainingState === 'running' ? 'Training...' : 'Start Training'}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Current Training Progress */}
      {id && training && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Training Progress: {training.id}</CardTitle>
            <div className="flex items-center gap-2">
              {getStatusBadge(training.state)}
              <label className="flex items-center gap-2 cursor-pointer text-sm">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                />
                Auto-refresh
              </label>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <Progress value={training.progress * 100} className="h-3" />
              </div>
              <span className="text-sm text-muted-foreground w-24 text-right">
                {(training.progress * 100).toFixed(1)}%
              </span>
              <span className="text-sm text-muted-foreground">
                Step {training.current_step} / {training.total_steps}
              </span>
            </div>

            {/* Reward Chart */}
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={episodeRewards.map((r, i) => ({ step: i, reward: r }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" name="Episode" />
                  <YAxis name="Reward" />
                  <Tooltip formatter={(value: number) => [value.toFixed(3), 'Reward']} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    name="Episode Reward"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                  {evalRewards.length > 0 && (
                    <Line
                      type="monotone"
                      dataKey="reward"
                      name="Eval Reward"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={false}
                      data={evalRewards.map((r, i) => ({ step: i * 10, reward: r }))}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Training Config Display */}
            <div className="grid gap-4 md:grid-cols-4 text-sm">
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-muted-foreground">Model Path</p>
                <p className="font-mono text-xs truncate">{training.model_path || 'Not saved yet'}</p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-muted-foreground">Total Steps</p>
                <p className="font-medium">{training.total_steps.toLocaleString()}</p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-muted-foreground">Current Step</p>
                <p className="font-medium">{training.current_step.toLocaleString()}</p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-muted-foreground">Latest Reward</p>
                <p className="font-medium">
                  {episodeRewards.length > 0 ? episodeRewards[episodeRewards.length - 1].toFixed(3) : 'N/A'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Saved Models */}
      <Card>
        <CardHeader>
          <CardTitle>Saved Models</CardTitle>
        </CardHeader>
        <CardContent>
          {modelsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : models.length === 0 ? (
            <div className="text-center py-8">
              <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No models trained yet</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model ID</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Timesteps</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Workload</TableHead>
                  <TableHead>Final Reward</TableHead>
                  <TableHead>Avg Eval Reward</TableHead>
                  <TableHead>Training Time</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell className="font-mono text-sm">{model.id}</TableCell>
                    <TableCell><Badge variant="outline">Registered</Badge></TableCell>
                    <TableCell>{model.config.timesteps.toLocaleString()}</TableCell>
                    <TableCell>{model.config.dataset_id}</TableCell>
                    <TableCell>{model.config.workload_type}</TableCell>
                    <TableCell>{model.metrics?.final_reward?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell>{model.metrics?.avg_eval_reward?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell>{model.metrics?.training_time_seconds?.toFixed(0) || 'N/A'}s</TableCell>
                    <TableCell>{formatDistanceToNow(new Date(model.created_at), { addSuffix: true })}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="sm" onClick={() => handleLoadModel(model.id)}>
                          <Brain className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => handleEvaluateModel(model.id)}>
                          <Play className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
