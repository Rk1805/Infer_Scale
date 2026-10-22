import { useState } from 'react'
import { useExperiments } from '../hooks/useData'
import { api, ExperimentJob } from '../lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Select } from '../components/ui/select'
import { Label } from '../components/ui/label'
import { Progress } from '../components/ui/progress'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/table'
import { useToast } from '../components/ui/toast'
import {
Download,
FileText,
Search,RotateCcw,
Eye,
Trash2,
ClipboardList
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

export function ExperimentHistory() {
  const { experiments, loading, refetch } = useExperiments()
  const { toast } = useToast()

  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentJob | null>(null)
  const [reportContent, setReportContent] = useState<string>('')

  const filteredExperiments = experiments.filter((exp) => {
    if (statusFilter === 'all') return true
    return exp.state === statusFilter
  })

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default">Completed</Badge>
      case 'running':
        return <Badge variant="secondary">Running</Badge>
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>
      case 'cancelled':
        return <Badge variant="outline">Cancelled</Badge>
      default:
        return <Badge variant="outline">Queued</Badge>
    }
  }

  const handleExport = async (exp: ExperimentJob, format: 'json' | 'csv') => {
    try {
      const blob = await api.exportExperiment(exp.id, format)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `experiment-${exp.id}.${format}`
      a.click()
      URL.revokeObjectURL(url)
      toast(`Exported as ${format.toUpperCase()}`, 'success')
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Export failed', 'destructive')
    }
  }

  const handleGenerateReport = async (exp: ExperimentJob) => {
    try {
      // Generate markdown report from experiment data
      const report = generateMarkdownReport(exp)
      setReportContent(report)
      setSelectedExperiment(exp)
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Report generation failed', 'destructive')
    }
  }

  const handleDownloadReport = () => {
    if (!selectedExperiment) return
    const blob = new Blob([reportContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report-${selectedExperiment.id}.md`
    a.click()
    URL.revokeObjectURL(url)
    toast('Report downloaded', 'success')
  }

  const handleDelete = async (exp: ExperimentJob) => {
    if (!confirm(`Delete experiment ${exp.id}?`)) return
    try {
      // TODO: Implement delete endpoint
      toast('Delete not yet implemented', 'destructive')
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Delete failed', 'destructive')
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Experiment History & Reports</h1>
          <p className="text-muted-foreground">View, export, and generate reports for past experiments</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={refetch} disabled={loading} className="gap-2">
            <RotateCcw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-0">
          <div className="flex flex-wrap items-center gap-4">
            <div className="space-y-1">
              <Label htmlFor="status-filter" className="text-sm">Status</Label>
              <Select
                id="status-filter"
                value={statusFilter}
                onValueChange={setStatusFilter}
              >
                <option value="all">All</option>
                <option value="completed">Completed</option>
                <option value="running">Running</option>
                <option value="failed">Failed</option>
                <option value="cancelled">Cancelled</option>
                <option value="queued">Queued</option>
              </Select>
            </div>
            <div className="flex-1" />
            <span className="text-sm text-muted-foreground">
              {filteredExperiments.length} of {experiments.length} experiments
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Experiment List */}
      <Card>
        <CardHeader>
          <CardTitle>Experiments</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Search className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : filteredExperiments.length === 0 ? (
            <div className="text-center py-8">
              <ClipboardList className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No experiments found</p>
              <Button onClick={() => window.location.href = '/experiment-studio'} className="mt-4">
                Create Experiment
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Policy</TableHead>
                  <TableHead>Workload</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Started</TableHead>
                  <TableHead>Completed</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredExperiments.map((exp) => (
                  <TableRow key={exp.id}>
                    <TableCell className="font-mono font-medium">{exp.id}</TableCell>
                    <TableCell>{getStatusBadge(exp.state)}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{exp.config.policy}</Badge>
                    </TableCell>
                    <TableCell>{exp.config.workload_type}</TableCell>
                    <TableCell>{exp.config.dataset_id}</TableCell>
                    <TableCell>
                      {exp.started_at ? formatDistanceToNow(new Date(exp.started_at), { addSuffix: true }) : '—'}
                    </TableCell>
                    <TableCell>
                      {exp.completed_at ? formatDistanceToNow(new Date(exp.completed_at), { addSuffix: true }) : '—'}
                    </TableCell>
                    <TableCell>
                      {exp.started_at && exp.completed_at
                        ? `${((new Date(exp.completed_at).getTime() - new Date(exp.started_at).getTime()) / 1000).toFixed(0)}s`
                        : '—'}
                    </TableCell>
                    <TableCell>
                      <Progress value={exp.progress * 100} className="h-2 w-32" />
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        {exp.state === 'completed' && exp.results && (
                          <>
                            <Button variant="ghost" size="sm" onClick={() => handleExport(exp, 'json')} title="Export JSON">
                              <Download className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => handleExport(exp, 'csv')} title="Export CSV">
                              <FileText className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => handleGenerateReport(exp)} title="Generate Report">
                              <ClipboardList className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => window.location.href = `/simulation/${exp.id}`} title="View Simulation">
                              <Eye className="h-3 w-3" />
                            </Button>
                          </>
                        )}
                        {exp.state === 'running' && (
                          <Button variant="ghost" size="sm" onClick={() => window.location.href = `/simulation/${exp.id}`} title="View Live">
                            <Eye className="h-3 w-3" />
                          </Button>
                        )}
                        <Button variant="ghost" size="sm" onClick={() => handleDelete(exp)} title="Delete">
                          <Trash2 className="h-3 w-3" />
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

      {/* Report Modal */}
      {selectedExperiment && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-background rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-lg font-bold">Experiment Report: {selectedExperiment.id}</h2>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={handleDownloadReport} className="gap-1">
                  <Download className="h-3 w-3" />
                  Download
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setSelectedExperiment(null)}>
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4 font-mono text-sm">
              <pre className="whitespace-pre-wrap">{reportContent}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function generateMarkdownReport(exp: ExperimentJob): string {
  const metrics = exp.results!
  const config = exp.config
  
  let md = `# Experiment Report: ${exp.id}\n\n`
  md += `**Generated:** ${new Date().toISOString()}\n\n`
  
  md += `## Configuration\n\n`
  md += `| Parameter | Value |\n`
  md += `|-----------|-------|\n`
  md += `| Policy | ${config.policy} |\n`
  md += `| Workload Type | ${config.workload_type} |\n`
  md += `| Dataset | ${config.dataset_id} |\n`
  md += `| Trace Window | ${config.trace_window_minutes} min |\n`
  md += `| Initial Replicas | ${config.initial_replicas} |\n`
  md += `| Min Replicas | ${config.min_replicas} |\n`
  md += `| Max Replicas | ${config.max_replicas} |\n`
  md += `| Replica Startup Delay | ${config.replica_startup_delay}s |\n`
  md += `| Scaling Cooldown | ${config.scaling_cooldown}s |\n`
  md += `| Decision Interval | ${config.decision_interval}s |\n`
  md += `| Episode Duration | ${config.episode_duration}s |\n`
  md += `| SLA Threshold | ${config.sla_threshold}s |\n`
  md += `| Random Seed | ${config.random_seed} |\n`
  md += `| Reward α (Cost) | ${config.reward_alpha} |\n`
  md += `| Reward β (SLA) | ${config.reward_beta} |\n`
  md += `| Reward γ (Queue) | ${config.reward_gamma} |\n`
  md += `| Reward δ (Churn) | ${config.reward_delta} |\n\n`
  
  md += `## Results\n\n`
  md += `| Metric | Value |\n`
  md += `|--------|-------|\n`
  md += `| Total Cost | $${metrics.total_cost.toFixed(2)} |\n`
  md += `| Avg Active Replicas | ${metrics.avg_active_replicas.toFixed(1)} |\n`
  md += `| p50 Latency | ${metrics.p50_latency_ms.toFixed(1)} ms |\n`
  md += `| p95 Latency | ${metrics.p95_latency_ms.toFixed(1)} ms |\n`
  md += `| p99 Latency | ${metrics.p99_latency_ms.toFixed(1)} ms |\n`
  md += `| SLA Violation Rate | ${(metrics.sla_violation_rate * 100).toFixed(2)}% |\n`
  md += `| Goodput | ${metrics.goodput.toFixed(2)} req/s |\n`
  md += `| Throughput | ${metrics.throughput.toFixed(2)} req/s |\n`
  md += `| Max Queue Depth | ${metrics.max_queue_depth} |\n`
  md += `| Avg Queue Depth | ${metrics.avg_queue_depth.toFixed(1)} |\n`
  md += `| Scaling Events | ${metrics.scaling_events} |\n`
  md += `| Burst Recovery Time | ${metrics.burst_recovery_time.toFixed(1)}s |\n`
  md += `| Unfinished Requests | ${metrics.unfinished_requests} |\n`
  md += `| Episode Reward | ${metrics.episode_reward.toFixed(3)} |\n\n`
  
  md += `## Timing\n\n`
  md += `- **Started:** ${exp.started_at || 'N/A'}\n`
  md += `- **Completed:** ${exp.completed_at || 'N/A'}\n`
  md += `- **Duration:** ${exp.started_at && exp.completed_at 
    ? ((new Date(exp.completed_at).getTime() - new Date(exp.started_at).getTime()) / 1000).toFixed(1) + 's'
    : 'N/A'}\n\n`
  
  if (exp.error) {
    md += `## Error\n\n`
    md += `\`\`\`\n${exp.error}\n\`\`\`\n\n`
  }
  
  md += `---\n*Report generated by InferScale*`
  
  return md
}
