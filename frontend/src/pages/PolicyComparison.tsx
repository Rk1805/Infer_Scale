import { useState, useEffect } from 'react'
import { useExperiments, useDatasets } from '../hooks/useData'
import { api, ExperimentConfig, ExperimentMetrics } from '../lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Select } from '../components/ui/select'
import { Label } from '../components/ui/label'
import { Badge } from '../components/ui/badge'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/table'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs'
import { Separator } from '../components/ui/separator'
import { useToast } from '../components/ui/toast'
import {
  Download,
  GitCompare,
  Loader2,
  Play,
  BarChart3,
  TrendingUp,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts'
import { formatDistanceToNow } from 'date-fns'

const POLICIES = [
  { value: 'fixed', label: 'Fixed Capacity' },
  { value: 'threshold', label: 'Threshold Autoscaling' },
  { value: 'queue', label: 'Queue-Based Autoscaling' },
  { value: 'ppo', label: 'PPO (RL)' },
]

export function PolicyComparison() {
  const { experiments, loading: expLoading, refetch: refetchExp } = useExperiments()
  const { datasets } = useDatasets()
  const { toast } = useToast()

  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([])
  const [runningComparison, setRunningComparison] = useState(false)
  const [comparisonResults, setComparisonResults] = useState<Record<string, ExperimentMetrics>>({})
  const [comparisonConfig, setComparisonConfig] = useState<Partial<ExperimentConfig>>({
    workload_type: 'steady',
    dataset_id: 'synthetic',
    trace_window_minutes: 30,
    initial_replicas: 2,
    min_replicas: 1,
    max_replicas: 8,
    replica_startup_delay: 10,
    scaling_cooldown: 30,
    decision_interval: 5,
    episode_duration: 3600,
    sla_threshold: 30,
    random_seed: 42,
  })

  // Filter completed experiments
  const completedExperiments = experiments.filter((e) => e.state === 'completed' && e.results)

  const handleRunComparison = async () => {
    setRunningComparison(true)
    const results: Record<string, ExperimentMetrics> = {}

    try {
      for (const policy of POLICIES.map((p) => p.value)) {
        const res = await api.createExperiment({
          ...comparisonConfig,
          policy,
        } as ExperimentConfig)

        // Wait for completion
        let job = await api.getExperiment(res.id)
        while (job.state === 'running' || job.state === 'queued') {
          await new Promise((r) => setTimeout(r, 2000))
          job = await api.getExperiment(res.id)
        }

        if (job.state === 'completed' && job.results) {
          results[policy] = job.results
        }
      }
      setComparisonResults(results)
      toast('Comparison completed', 'success')
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Comparison failed', 'destructive')
    } finally {
      setRunningComparison(false)
    }
  }

  const handleExport = (format: 'json' | 'csv') => {
    if (format === 'json') {
      const blob = new Blob([JSON.stringify(comparisonResults, null, 2)], { type: 'application/json' })
      downloadBlob(blob, 'policy-comparison.json')
    } else {
      const headers = ['Policy', 'Total Cost', 'Avg Replicas', 'p50 Latency (ms)', 'p95 Latency (ms)', 'p99 Latency (ms)', 'SLA Violation Rate', 'Goodput', 'Throughput', 'Max Queue Depth', 'Avg Queue Depth', 'Scaling Events']
      const rows = Object.entries(comparisonResults).map(([policy, m]) => [
        policy,
        m.total_cost.toFixed(2),
        m.avg_active_replicas.toFixed(1),
        m.p50_latency_ms.toFixed(1),
        m.p95_latency_ms.toFixed(1),
        m.p99_latency_ms.toFixed(1),
        (m.sla_violation_rate * 100).toFixed(2),
        m.goodput.toFixed(2),
        m.throughput.toFixed(2),
        m.max_queue_depth.toString(),
        m.avg_queue_depth.toFixed(1),
        m.scaling_events.toString(),
      ])
      const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
      const blob = new Blob([csv], { type: 'text/csv' })
      downloadBlob(blob, 'policy-comparison.csv')
    }
    toast(`Exported as ${format.toUpperCase()}`, 'success')
  }

  function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const policiesWithResults = POLICIES.map((p) => ({
    ...p,
    metrics: comparisonResults[p.value],
  }))

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Policy Comparison</h1>
          <p className="text-muted-foreground">Compare autoscaling policies on the same workload</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={handleRunComparison} disabled={runningComparison} className="gap-2">
            <GitCompare className="h-4 w-4" />
            {runningComparison ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Run Comparison'}
          </Button>
          <Button variant="outline" onClick={() => handleExport('json')} disabled={Object.keys(comparisonResults).length === 0} className="gap-2">
            <Download className="h-4 w-4" />
            Export JSON
          </Button>
          <Button variant="outline" onClick={() => handleExport('csv')} disabled={Object.keys(comparisonResults).length === 0} className="gap-2">
            <Download className="h-4 w-4" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Comparison Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="comp_workload_type">Workload Type</Label>
              <Select
                id="comp_workload_type"
                value={comparisonConfig.workload_type || 'steady'}
                onValueChange={(v) => setComparisonConfig((prev) => ({ ...prev, workload_type: v }))}
              >
                <option value="steady">Steady</option>
                <option value="periodic">Periodic</option>
                <option value="flash_crowd">Flash Crowd</option>
                <option value="mixed">Mixed</option>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="comp_dataset_id">Dataset</Label>
              <Select
                id="comp_dataset_id"
                value={comparisonConfig.dataset_id || 'synthetic'}
                onValueChange={(v) => setComparisonConfig((prev) => ({ ...prev, dataset_id: v }))}
              >
                <option value="synthetic">Synthetic</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="comp_episode_duration">Episode Duration (s)</Label>
              <Select
                id="comp_episode_duration"
                value={comparisonConfig.episode_duration?.toString() || '3600'}
                onValueChange={(v) => setComparisonConfig((prev) => ({ ...prev, episode_duration: parseFloat(v) }))}
              >
                <option value="300">5 min (short)</option>
                <option value="1800">30 min</option>
                <option value="3600">1 hour</option>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="comp_sla_threshold">SLA Threshold (s)</Label>
              <Select
                id="comp_sla_threshold"
                value={comparisonConfig.sla_threshold?.toString() || '30'}
                onValueChange={(v) => setComparisonConfig((prev) => ({ ...prev, sla_threshold: parseFloat(v) }))}
              >
                <option value="10">10s</option>
                <option value="30">30s</option>
                <option value="60">60s</option>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {Object.keys(comparisonResults).length > 0 && (
        <>
          {/* Summary Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            {POLICIES.map((policy) => {
              const m = comparisonResults[policy.value]
              if (!m) return null
              return (
                <Card key={policy.value} className="relative">
                  <div className="absolute top-2 right-2">
                    <Badge variant="outline">{policy.label}</Badge>
                  </div>
                  <CardContent className="py-4">
                    <div className="space-y-2">
                      <MetricRow label="Cost" value={`$${m.total_cost.toFixed(2)}`} />
                      <MetricRow label="Avg Replicas" value={m.avg_active_replicas.toFixed(1)} />
                      <MetricRow label="p95 Latency" value={`${m.p95_latency_ms.toFixed(1)}ms`} />
                      <MetricRow label="SLA Violations" value={`${(m.sla_violation_rate * 100).toFixed(1)}%`} />
                      <MetricRow label="Goodput" value={`${m.goodput.toFixed(2)} req/s`} />
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Detailed Comparison Table */}
          <Card>
            <CardHeader>
              <CardTitle>Detailed Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Policy</TableHead>
                    <TableHead>Total Cost</TableHead>
                    <TableHead>Avg Replicas</TableHead>
                    <TableHead>p50 Latency</TableHead>
                    <TableHead>p95 Latency</TableHead>
                    <TableHead>p99 Latency</TableHead>
                    <TableHead>SLA Violations</TableHead>
                    <TableHead>Goodput</TableHead>
                    <TableHead>Throughput</TableHead>
                    <TableHead>Max Queue</TableHead>
                    <TableHead>Avg Queue</TableHead>
                    <TableHead>Scaling Events</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {POLICIES.map((policy) => {
                    const m = comparisonResults[policy.value]
                    if (!m) return null
                    return (
                      <TableRow key={policy.value}>
                        <TableCell>
                          <Badge variant="outline">{policy.label}</Badge>
                        </TableCell>
                        <TableCell>${m.total_cost.toFixed(2)}</TableCell>
                        <TableCell>{m.avg_active_replicas.toFixed(1)}</TableCell>
                        <TableCell>{m.p50_latency_ms.toFixed(1)}ms</TableCell>
                        <TableCell>{m.p95_latency_ms.toFixed(1)}ms</TableCell>
                        <TableCell>{m.p99_latency_ms.toFixed(1)}ms</TableCell>
                        <TableCell>{(m.sla_violation_rate * 100).toFixed(1)}%</TableCell>
                        <TableCell>{m.goodput.toFixed(2)}</TableCell>
                        <TableCell>{m.throughput.toFixed(2)}</TableCell>
                        <TableCell>{m.max_queue_depth}</TableCell>
                        <TableCell>{m.avg_queue_depth.toFixed(1)}</TableCell>
                        <TableCell>{m.scaling_events}</TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Charts */}
          <Tabs defaultValue="cost-latency" className="space-y-4">
            <TabsList>
              <TabsTrigger value="cost-latency">Cost vs Latency</TabsTrigger>
              <TabsTrigger value="replicas">Replica Usage</TabsTrigger>
              <TabsTrigger value="latency-dist">Latency Distribution</TabsTrigger>
              <TabsTrigger value="sla">SLA Compliance</TabsTrigger>
            </TabsList>

            <TabsContent value="cost-latency">
              <Card>
                <CardHeader>
                  <CardTitle>Cost vs p95 Latency Trade-off</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={350}>
                    <ScatterChart>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="cost" name="Total Cost ($)" />
                      <YAxis type="number" dataKey="latency" name="p95 Latency (ms)" />
                      <Tooltip
                        formatter={(value: number, name: string) => [
                          name === 'cost' ? `$${value.toFixed(2)}` : `${value.toFixed(1)}ms`,
                          name,
                        ]}
                        labelFormatter={(_, payload) => payload[0]?.payload?.policy || ''}
                      />
                      <Legend />
                      <Scatter
                        name="Policies"
                        data={POLICIES.map((p) => {
                          const m = comparisonResults[p.value]
                          return m ? { policy: p.label, cost: m.total_cost, latency: m.p95_latency_ms } : null
                        }).filter(Boolean) as any[]}
                        fill="#3b82f6"
                        stroke="#3b82f6"
                        size={8}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="replicas">
              <Card>
                <CardHeader>
                  <CardTitle>Average Active Replicas</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={POLICIES.map((p) => {
                      const m = comparisonResults[p.value]
                      return m ? { policy: p.label, replicas: m.avg_active_replicas } : null
                    }).filter(Boolean) as any[]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="policy" name="Policy" />
                      <YAxis name="Avg Active Replicas" />
                      <Tooltip />
                      <Bar dataKey="replicas" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="latency-dist">
              <Card>
                <CardHeader>
                  <CardTitle>Latency Percentiles by Policy</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={POLICIES.map((p) => {
                      const m = comparisonResults[p.value]
                      return m ? { policy: p.label, p50: m.p50_latency_ms, p95: m.p95_latency_ms, p99: m.p99_latency_ms } : null
                    }).filter(Boolean) as any[]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="policy" name="Policy" />
                      <YAxis name="Latency (ms)" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="p50" name="p50" fill="#22c55e" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="p95" name="p95" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="p99" name="p99" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="sla">
              <Card>
                <CardHeader>
                  <CardTitle>SLA Violation Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={POLICIES.map((p) => {
                      const m = comparisonResults[p.value]
                      return m ? { policy: p.label, violations: m.sla_violation_rate * 100 } : null
                    }).filter(Boolean) as any[]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="policy" name="Policy" />
                      <YAxis name="SLA Violation Rate (%)" />
                      <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Violations']} />
                      <Bar dataKey="violations" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Existing Experiment Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Compare Existing Experiments</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Select completed experiments to compare their results directly
              </p>
              <div className="flex flex-wrap gap-2 mb-4">
                {completedExperiments.slice(0, 20).map((exp) => (
                  <Label key={exp.id} className="cursor-pointer flex items-center gap-2 px-3 py-1 rounded border hover:bg-accent">
                    <input
                      type="checkbox"
                      checked={selectedExperiments.includes(exp.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedExperiments([...selectedExperiments, exp.id])
                        } else {
                          setSelectedExperiments(selectedExperiments.filter((id) => id !== exp.id))
                        }
                      }}
                      className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                    />
                    <span className="text-sm">{exp.id} ({exp.config.policy})</span>
                    <Badge variant="secondary" className="ml-1">{formatDistanceToNow(new Date(exp.completed_at!), { addSuffix: true })}</Badge>
                  </Label>
                ))}
              </div>
              {selectedExperiments.length > 0 && (
                <Button variant="outline" onClick={() => compareSelected()} className="gap-2">
                  <GitCompare className="h-4 w-4" />
                  Compare Selected ({selectedExperiments.length})
                </Button>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {Object.keys(comparisonResults).length === 0 && (
        <Card>
          <CardContent className="py-12 text-center">
            <GitCompare className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Comparison Results</h3>
            <p className="text-muted-foreground mb-4">
              Configure the comparison parameters above and click "Run Comparison" to evaluate all four policies
              on the same workload trace.
            </p>
            <Button onClick={handleRunComparison} disabled={runningComparison} className="gap-2">
              <Play className="h-4 w-4" />
              Run Comparison
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}

async function compareSelected() {
  // TODO: Implement comparison of selected existing experiments
  alert('Feature coming soon: compare selected experiments')
}