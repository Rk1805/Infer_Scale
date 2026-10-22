import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useExperiment, useExperimentMetrics, useExperimentTimeline } from '../hooks/useData'
import { api, TimelineEvent } from '../lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs'
import { Progress } from '../components/ui/progress'
import { useToast } from '../components/ui/toast'
import { Label } from '../components/ui/label'
import { Select } from '../components/ui/select'
import { Server } from 'lucide-react'
import {
Play,RotateCcw,
Download,Loader2,
CheckCircle,
AlertCircle,
XCircle
} from 'lucide-react'
import {
Line,
AreaChart,
Area,Bar,
XAxis,
YAxis,
CartesianGrid,
Tooltip,
Legend,
ResponsiveContainer,
ComposedChart
} from 'recharts'
import { formatDistanceToNow } from 'date-fns'

export function Simulation() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { experiment, loading: expLoading, error: expError, refetch: refetchExp } = useExperiment(id || null)
  const { metrics } = useExperimentMetrics(id || null)
  const { timeline } = useExperimentTimeline(id || null)
  const { toast } = useToast()

  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [timeRange, setTimeRange] = useState<[number, number]>([0, 3600])
  const [selectedStep, setSelectedStep] = useState<TimelineEvent | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Auto-refresh for running experiments
  useEffect(() => {
    if (!experiment || !autoRefresh) return
    if (experiment.state === 'running') {
      const interval = setInterval(() => refetchExp(), 2000)
      return () => clearInterval(interval)
    }
  }, [experiment, autoRefresh, refetchExp])

  const handleControl = async (action: string) => {
    if (!experiment) return
    try {
      if (action === 'cancel') {
        await api.cancelExperiment(experiment.id)
        toast('Experiment cancelled', 'success')
        refetchExp()
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

  if (!id) {
    return (
      <div className="p-6 text-center">
        <h1 className="text-3xl font-bold mb-4">Simulation View</h1>
        <p className="text-muted-foreground">Select an experiment from the history or launch a new one</p>
        <Button onClick={() => navigate('/experiment-studio')} className="mt-4">
          <Play className="mr-2 h-4 w-4" />
          Launch New Experiment
        </Button>
      </div>
    )
  }

  if (expLoading) {
    return (
      <div className="p-6 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (expError || !experiment) {
    return (
      <div className="p-6 text-center">
        <AlertCircle className="h-12 w-12 text-destructive mx-auto mb-4" />
        <h2 className="text-xl font-bold">Experiment Not Found</h2>
        <p className="text-muted-foreground mt-2">{expError || 'Unknown error'}</p>
        <Button onClick={() => navigate('/experiment-history')} className="mt-4">
          View History
        </Button>
      </div>
    )
  }

  // Process timeline data for charts
  const requests = timeline.filter((e) => e.type === 'request')
  const actions = timeline.filter((e) => e.type === 'action')

  // Time series data for charts
  const timeSeriesData = generateTimeSeries(requests, actions, timeRange[1])

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold">Simulation: {experiment.id}</h1>
          <p className="text-muted-foreground">
            {experiment.config.policy} • {experiment.config.workload_type} • {experiment.config.dataset_id}
          </p>
        </div>
        <div className="flex items-center gap-4">
          {getStatusBadge(experiment.state)}
          <div className="flex items-center gap-2">
            <Label className="text-sm">Speed</Label>
            <Select value={playbackSpeed.toString()} onValueChange={(v) => setPlaybackSpeed(parseFloat(v))}>
              <option value="0.5">0.5x</option>
              <option value="1">1x</option>
              <option value="2">2x</option>
              <option value="5">5x</option>
              <option value="10">10x</option>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
            />
            <Label className="text-sm cursor-pointer">Auto-refresh</Label>
          </div>
          {experiment.state === 'running' && (
            <Button variant="outline" onClick={() => handleControl('cancel')} className="gap-2">
              <XCircle className="h-4 w-4" />
              Cancel
            </Button>
          )}
          <Button variant="outline" onClick={() => navigate('/experiment-history')} className="gap-2">
            <RotateCcw className="h-4 w-4" />
            Back
          </Button>
        </div>
      </div>

      {/* Progress Bar */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Progress value={experiment.progress * 100} className="h-3" />
            </div>
            <span className="text-sm text-muted-foreground w-24 text-right">
              {(experiment.progress * 100).toFixed(1)}%
            </span>
            <span className="text-sm text-muted-foreground w-32 text-right">
              {experiment.state === 'running' && experiment.started_at
                ? formatDistanceToNow(new Date(experiment.started_at), { addSuffix: true })
                : 'Not started'}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <MetricCard title="Active Replicas" value={metrics?.avg_active_replicas?.toFixed(1) || '0'} icon={<Server className="h-5 w-5" />} />
        <MetricCard title="p50 Latency" value={`${(metrics?.p50_latency_ms || 0).toFixed(1)}ms`} icon={<RotateCcw className="h-5 w-5" />} />
        <MetricCard title="p95 Latency" value={`${(metrics?.p95_latency_ms || 0).toFixed(1)}ms`} icon={<RotateCcw className="h-5 w-5" />} />
        <MetricCard title="SLA Violations" value={`${((metrics?.sla_violation_rate || 0) * 100).toFixed(1)}%`} icon={<AlertCircle className="h-5 w-5" />} />
        <MetricCard title="Total Cost" value={`$${(metrics?.total_cost || 0).toFixed(2)}`} icon={<Download className="h-5 w-5" />} />
      </div>

      {/* Charts */}
      <Tabs defaultValue="latency" className="space-y-4">
        <TabsList>
          <TabsTrigger value="latency">Latency</TabsTrigger>
          <TabsTrigger value="queue">Queue Depth</TabsTrigger>
          <TabsTrigger value="replicas">Replicas</TabsTrigger>
          <TabsTrigger value="throughput">Throughput</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>

        <TabsContent value="latency">
          <Card>
            <CardHeader>
              <CardTitle>Latency Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={timeSeriesData.latency}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" label="Time (s)" />
                  <YAxis label="Latency (ms)" />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="p50" name="p50" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} />
                  <Area type="monotone" dataKey="p95" name="p95" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} />
                  <Area type="monotone" dataKey="p99" name="p99" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} />
                  <Line type="monotone" dataKey="sla" name="SLA Threshold" stroke="#6b7280" strokeDasharray="5 5" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="queue">
          <Card>
            <CardHeader>
              <CardTitle>Queue Depth Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={timeSeriesData.queue}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" label="Time (s)" />
                  <YAxis label="Queue Depth" />
                  <Tooltip />
                  <Area type="monotone" dataKey="depth" name="Queue Depth" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.1} />
                  <Area type="monotone" dataKey="waiting" name="Waiting Requests" stroke="#ec4899" fill="#ec4899" fillOpacity={0.1} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="replicas">
          <Card>
            <CardHeader>
              <CardTitle>Active Replicas Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={timeSeriesData.replicas}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" label="Time (s)" />
                  <YAxis label="Replicas" />
                  <Tooltip />
                  <Area type="monotone" dataKey="active" name="Active" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} />
                  <Area type="monotone" dataKey="starting" name="Starting" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} />
                  <Area type="monotone" dataKey="total" name="Total" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.05} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="throughput">
          <Card>
            <CardHeader>
              <CardTitle>Throughput & Goodput</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={timeSeriesData.throughput}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" label="Time (s)" />
                  <YAxis label="Requests/s" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="throughput" name="Throughput" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="goodput" name="Goodput" fill="#22c55e" radius={[4, 4, 0, 0]} />
                  <Line type="monotone" dataKey="arrivals" name="Arrivals" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeline">
          <Card>
            <CardHeader>
              <CardTitle>Event Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <TimelineView
                requests={requests}
                actions={actions}
                selectedStep={selectedStep}
                onSelectStep={setSelectedStep}
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Step Inspector */}
      {selectedStep && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Step Details</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => setSelectedStep(null)}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <pre className="text-sm bg-muted p-4 rounded max-h-60 overflow-auto">
              {JSON.stringify(selectedStep, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* Detailed Metrics */}
      {metrics && (
        <Card>
          <CardHeader>
            <CardTitle>Detailed Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <MetricGroup title="Latency" items={[
                { label: 'p50', value: `${metrics.p50_latency_ms.toFixed(1)}ms` },
                { label: 'p95', value: `${metrics.p95_latency_ms.toFixed(1)}ms` },
                { label: 'p99', value: `${metrics.p99_latency_ms.toFixed(1)}ms` },
                { label: 'Avg Queueing Delay', value: `${metrics.avg_queueing_delay_ms.toFixed(1)}ms` },
                { label: 'Avg Service Time', value: `${metrics.avg_service_time_ms.toFixed(1)}ms` },
              ]} />
              <MetricGroup title="Throughput" items={[
                { label: 'Throughput', value: `${metrics.throughput.toFixed(2)} req/s` },
                { label: 'Goodput', value: `${metrics.goodput.toFixed(2)} req/s` },
                { label: 'Completions', value: metrics.completions?.toString() || '0' },
                { label: 'Arrivals', value: metrics.arrivals?.toString() || '0' },
                { label: 'Unfinished', value: metrics.unfinished_requests?.toString() || '0' },
              ]} />
              <MetricGroup title="Scaling" items={[
                { label: 'Avg Active Replicas', value: metrics.avg_active_replicas?.toFixed(1) || '0' },
                { label: 'Max Queue Depth', value: metrics.max_queue_depth?.toString() || '0' },
                { label: 'Avg Queue Depth', value: metrics.avg_queue_depth?.toFixed(1) || '0' },
                { label: 'Scaling Events', value: metrics.scaling_events?.toString() || '0' },
                { label: 'Burst Recovery', value: `${(metrics.burst_recovery_time || 0).toFixed(1)}s` },
              ]} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Export Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Export Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button variant="outline" onClick={() => exportResults('json')} className="gap-2">
              <Download className="h-4 w-4" />
              Export JSON
            </Button>
            <Button variant="outline" onClick={() => exportResults('csv')} className="gap-2">
              <Download className="h-4 w-4" />
              Export CSV
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  async function exportResults(format: 'json' | 'csv') {
    if (!experiment) return
    try {
      const blob = await api.exportExperiment(experiment.id, format)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `experiment-${experiment.id}.${format}`
      a.click()
      URL.revokeObjectURL(url)
      toast(`Exported as ${format.toUpperCase()}`, 'success')
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Export failed', 'destructive')
    }
  }
}

function MetricCard({ title, value, icon }: { title: string; value: string; icon: React.ReactNode }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-muted">{icon}</div>
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function MetricGroup({ title, items }: { title: string; items: { label: string; value: string }[] }) {
  return (
    <div className="space-y-3 p-4 bg-muted/50 rounded-lg">
      <h4 className="font-medium text-sm text-muted-foreground">{title}</h4>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item.label} className="flex justify-between text-sm">
            <span className="text-muted-foreground">{item.label}</span>
            <span className="font-medium">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function generateTimeSeries(
  requests: TimelineEvent[],
  actions: TimelineEvent[],
  maxTime: number
) {
  const interval = 30 // 30 second buckets
  const buckets = Math.ceil(maxTime / interval)
  
  const latencyData = []
  const queueData = []
  const replicasData = []
  const throughputData = []
  
  let activeReplicas = 2
  const startingReplicas = 0
  
  for (let i = 0; i <= buckets; i++) {
    const t = i * interval
    const bucketRequests = requests.filter((r) => 
      r.arrival !== undefined && r.arrival >= t - interval && r.arrival < t
    )
    const bucketCompleted = requests.filter((r) => 
      r.decode_end !== undefined && r.decode_end >= t - interval && r.decode_end < t
    )
    const bucketActions = actions.filter((a) => 
      a.time !== undefined && a.time >= t - interval && a.time < t
    )
    
    // Update replica count from actions
    bucketActions.forEach((a) => {
      if (a.replicas !== undefined) {
        activeReplicas = a.replicas
      }
    })
    
    const latencies = bucketCompleted
      .filter((r) => r.decode_end !== undefined && r.arrival !== undefined)
      .map((r) => (r.decode_end! - r.arrival!) * 1000)
    
    const p50 = latencies.length ? percentile(latencies, 50) : 0
    const p95 = latencies.length ? percentile(latencies, 95) : 0
    const p99 = latencies.length ? percentile(latencies, 99) : 0
    
    latencyData.push({
      time: t,
      p50,
      p95,
      p99,
      sla: 30000, // 30s SLA in ms
    })
    
    queueData.push({
      time: t,
      depth: bucketRequests.length,
      waiting: bucketRequests.length,
    })
    
    replicasData.push({
      time: t,
      active: activeReplicas,
      starting: startingReplicas,
      total: activeReplicas + startingReplicas,
    })
    
    throughputData.push({
      time: t,
      throughput: bucketCompleted.length / (interval / 1),
      goodput: bucketCompleted.filter((r) => !r.sla_violated).length / (interval / 1),
      arrivals: bucketRequests.length / (interval / 1),
    })
  }
  
  return {
    latency: latencyData,
    queue: queueData,
    replicas: replicasData,
    throughput: throughputData,
  }
}

function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  const idx = Math.ceil(p / 100 * sorted.length) - 1
  return sorted[Math.max(0, idx)]
}

function TimelineView({
  requests,
  actions,
  onSelectStep,
  timeRange,
  onTimeRangeChange,
}: {
  requests: TimelineEvent[]
  actions: TimelineEvent[]
  selectedStep: TimelineEvent | null
  onSelectStep: (step: TimelineEvent | null) => void
  timeRange: [number, number]
  onTimeRangeChange: (range: [number, number]) => void
}) {
  const filteredRequests = requests.filter((r) => 
    r.arrival !== undefined && r.arrival >= timeRange[0] && r.arrival <= timeRange[1]
  )
  const filteredActions = actions.filter((a) => 
    a.time !== undefined && a.time >= timeRange[0] && a.time <= timeRange[1]
  )

  return (
    <div className="space-y-4">
      {/* Time range selector */}
      <div className="flex items-center gap-4">
        <Label className="text-sm">Time Range:</Label>
        <input
          type="range"
          min={0}
          max={3600}
          value={timeRange[0]}
          onChange={(e) => onTimeRangeChange([parseFloat(e.target.value), timeRange[1]])}
          className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer"
        />
        <span className="text-sm text-muted-foreground w-20">
          {timeRange[0]}s - {timeRange[1]}s
        </span>
        <input
          type="range"
          min={0}
          max={3600}
          value={timeRange[1]}
          onChange={(e) => onTimeRangeChange([timeRange[0], parseFloat(e.target.value)])}
          className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Timeline visualization */}
      <div className="relative h-64 border rounded-lg bg-background overflow-x-auto">
        <svg width="100%" height="100%" className="absolute inset-0">
          {/* Time axis */}
          <line x1={50} y1={20} x2={50} y2={580} stroke="currentColor" strokeOpacity={0.2} />
          
          {/* Requests as horizontal bars */}
          {filteredRequests.map((r, i) => {
            if (!r.arrival || !r.decode_end) return null
            const x1 = 50 + (r.arrival - timeRange[0]) / (timeRange[1] - timeRange[0]) * 800
            const x2 = 50 + (r.decode_end - timeRange[0]) / (timeRange[1] - timeRange[0]) * 800
            const y = 40 + (i % 20) * 25
            const isViolated = r.sla_violated
            
            return (
              <g key={r.id} onClick={() => onSelectStep(r)} style={{ cursor: 'pointer' }}>
                <line
                  x1={x1}
                  y1={y}
                  x2={x2}
                  y2={y}
                  stroke={isViolated ? '#ef4444' : '#3b82f6'}
                  strokeWidth={3}
                  strokeLinecap="round"
                />
                {r.prefill_start && r.prefill_end && (
                  <line
                    x1={50 + (r.prefill_start - timeRange[0]) / (timeRange[1] - timeRange[0]) * 800}
                    y1={y}
                    x2={50 + (r.prefill_end - timeRange[0]) / (timeRange[1] - timeRange[0]) * 800}
                    y2={y}
                    stroke="#f59e0b"
                    strokeWidth={2}
                    strokeLinecap="round"
                    opacity={0.7}
                  />
                )}
              </g>
            )
          })}
          
          {/* Scaling actions as markers */}
          {filteredActions.map((a) => {
            if (!a.time) return null
            const x = 50 + (a.time - timeRange[0]) / (timeRange[1] - timeRange[0]) * 800
            const color = a.action === 'scale_up' ? '#22c55e' : a.action === 'scale_down' ? '#ef4444' : '#6b7280'
            
            return (
              <g key={`action-${a.time}`} onClick={() => onSelectStep(a)} style={{ cursor: 'pointer' }}>
                <polygon
                  points={`${x},30 ${x-6},42 ${x+6},42`}
                  fill={color}
                  opacity={0.8}
                />
                <text x={x} y={25} textAnchor="middle" fontSize="10" fill={color}>
                  {a.action === 'scale_up' ? '▲' : a.action === 'scale_down' ? '▼' : '●'}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-blue-500 rounded" /> Request</span>
        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-amber-500 rounded" /> Prefill</span>
        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-red-500 rounded" /> SLA Violation</span>
        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-green-500 rounded" /> Scale Up</span>
        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-red-500 rounded" /> Scale Down</span>
        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-gray-500 rounded" /> Hold</span>
      </div>
    </div>
  )
}
