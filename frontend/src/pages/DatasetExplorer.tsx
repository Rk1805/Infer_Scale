import { useState } from 'react'
import { useDatasets, useDatasetStats } from '../hooks/useData'
import { api, DatasetStats } from '../lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/table'
import { Separator } from '../components/ui/separator'
import { Progress } from '../components/ui/progress'
import { useToast } from '../components/ui/toast'
import {
  Database,
  Download,
  Upload,
  Search,
  Filter,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

export function DatasetExplorer() {
  const { datasets, loading, refetch } = useDatasets()
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const { stats, loading: statsLoading } = useDatasetStats(selectedDataset)
  const { toast } = useToast()

  const handleValidate = async (file: File) => {
    try {
      const result = await api.validateDataset(file)
      if (result.valid) {
        toast(`Validated ${result.record_count} records`, 'success')
      } else {
        toast(`Validation failed: ${result.errors.join(', ')}`, 'destructive')
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Validation failed', 'destructive')
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleValidate(file)
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dataset Explorer</h1>
          <p className="text-muted-foreground">Explore and validate workload datasets</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <Upload className="h-4 w-4" />
            <span className="text-sm">Upload CSV</span>
            <input
              type="file"
              accept=".csv,.json"
              onChange={handleFileChange}
              className="sr-only"
            />
          </label>
          <Button variant="outline" onClick={refetch} disabled={loading} className="gap-2">
            <Database className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Dataset List */}
      <Card>
        <CardHeader>
          <CardTitle>Available Datasets</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Search className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : datasets.length === 0 ? (
            <div className="text-center py-8">
              <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No datasets available</p>
              <p className="text-sm text-muted-foreground mt-2">
                Synthetic workloads are generated on-demand. Upload a CSV or JSON file to add a custom dataset.
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Records</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Hash</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.map((dataset) => (
                  <TableRow key={dataset.id}>
                    <TableCell className="font-medium">{dataset.name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{dataset.source}</Badge>
                    </TableCell>
                    <TableCell>{dataset.record_count.toLocaleString()}</TableCell>
                    <TableCell>{(dataset.duration_seconds / 60).toFixed(1)} min</TableCell>
                    <TableCell className="font-mono text-xs">{dataset.hash.slice(0, 16)}...</TableCell>
                    <TableCell>{new Date(dataset.created_at).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedDataset(dataset.id)}
                        className="gap-1"
                      >
                        <Filter className="h-3 w-3" />
                        Explore
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Dataset Details */}
      {selectedDataset && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Dataset Details: {selectedDataset}</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => setSelectedDataset(null)}>
              <Search className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Search className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : stats ? (
              <DatasetDetails stats={stats} onExport={() => exportDataset(selectedDataset!)} />
            ) : (
              <p className="text-muted-foreground">Failed to load dataset stats</p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function DatasetDetails({ stats, onExport }: { stats: DatasetStats; onExport: () => void }) {
  return (
    <Tabs defaultValue="overview" className="space-y-4">
      <TabsList>
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="tokens">Token Distribution</TabsTrigger>
        <TabsTrigger value="arrivals">Arrival Rate</TabsTrigger>
        <TabsTrigger value="bursts">Burst Detection</TabsTrigger>
        <TabsTrigger value="quality">Data Quality</TabsTrigger>
      </TabsList>

      <TabsContent value="overview">
        <div className="grid gap-4 md:grid-cols-4">
          <StatCard title="Records" value={stats.record_count.toLocaleString()} icon={<Database className="h-5 w-5" />} />
          <StatCard title="Duration" value={`${(stats.duration_seconds / 60).toFixed(1)} min`} icon={<Database className="h-5 w-5" />} />
          <StatCard title="Avg Arrival Rate" value={`${(stats.record_count / stats.duration_seconds).toFixed(2)}/s`} icon={<Database className="h-5 w-5" />} />
          <StatCard title="Missing Values" value={stats.missing_values.toString()} icon={<AlertTriangle className="h-5 w-5" />} />
        </div>

        <Separator className="my-4" />

        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-3">
            <h4 className="font-medium">Prompt Tokens</h4>
            <TokenStatsTable stats={stats.prompt_tokens} />
          </div>
          <div className="space-y-3">
            <h4 className="font-medium">Output Tokens</h4>
            <TokenStatsTable stats={stats.output_tokens} />
          </div>
        </div>

        <Separator className="my-4" />

        <div className="flex justify-end">
          <Button variant="outline" onClick={onExport} className="gap-2">
            <Download className="h-4 w-4" />
            Export Stats
          </Button>
        </div>
      </TabsContent>

      <TabsContent value="tokens">
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Prompt Token Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={generateTokenHistogram(stats.prompt_tokens)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" name="Token Range" />
                  <YAxis name="Count" />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Output Token Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={generateTokenHistogram(stats.output_tokens)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" name="Token Range" />
                  <YAxis name="Count" />
                  <Tooltip />
                  <Bar dataKey="count" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </TabsContent>

      <TabsContent value="arrivals">
        <Card>
          <CardHeader>
            <CardTitle>Arrival Rate Time Series</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={stats.arrival_rate.slice(-200)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" name="Time" tickFormatter={(v) => new Date(v * 1000).toLocaleTimeString()} />
                <YAxis name="Requests/sec" />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(2), 'Rate']}
                  labelFormatter={(v) => new Date(v * 1000).toLocaleTimeString()}
                />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="bursts">
        <Card>
          <CardHeader>
            <CardTitle>Detected Bursts</CardTitle>
          </CardHeader>
          <CardContent>
            {stats.burst_detection.length === 0 ? (
              <p className="text-muted-foreground text-center py-8">No bursts detected</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Start Time</TableHead>
                    <TableHead>End Time</TableHead>
                    <TableHead>Duration (s)</TableHead>
                    <TableHead>Peak Rate</TableHead>
                    <TableHead>Avg Rate</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stats.burst_detection.map((burst, i) => (
                    <TableRow key={i}>
                      <TableCell>{new Date(burst.start_time * 1000).toLocaleTimeString()}</TableCell>
                      <TableCell>{new Date(burst.end_time * 1000).toLocaleTimeString()}</TableCell>
                      <TableCell>{burst.duration.toFixed(1)}</TableCell>
                      <TableCell>{burst.peak_rate.toFixed(2)}/s</TableCell>
                      <TableCell>{burst.avg_rate.toFixed(2)}/s</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="quality">
        <div className="grid gap-4 md:grid-cols-3">
          <StatCard
            title="Missing Values"
            value={stats.missing_values.toString()}
            icon={<AlertTriangle className="h-5 w-5 text-yellow-500" />}
            description="Records with missing fields"
          />
          <StatCard
            title="Invalid Rows"
            value={stats.invalid_rows.toString()}
            icon={<AlertTriangle className="h-5 w-5 text-red-500" />}
            description="Records failing validation"
          />
          <StatCard
            title="Validity Rate"
            value={`${((stats.record_count - stats.invalid_rows) / stats.record_count * 100).toFixed(1)}%`}
            icon={<CheckCircle className="h-5 w-5 text-green-500" />}
            description="Percentage of valid records"
          />
        </div>

        <Separator className="my-4" />

        <Card>
          <CardHeader>
            <CardTitle>Schema</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Field</TableHead>
                  <TableHead>Type</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(stats.schema).map(([field, type]) => (
                  <TableRow key={field}>
                    <TableCell className="font-mono">{field}</TableCell>
                    <TableCell>{type}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}

function StatCard({ title, value, icon, description }: { title: string; value: string; icon: React.ReactNode; description?: string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-muted">{icon}</div>
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
            {description && <p className="text-xs text-muted-foreground">{description}</p>}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function TokenStatsTable({ stats }: { stats: any }) {
  return (
    <Table>
      <TableBody>
        <TableRow>
          <TableCell className="text-muted-foreground">Min</TableCell>
          <TableCell>{stats.min}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">Max</TableCell>
          <TableCell>{stats.max}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">Mean</TableCell>
          <TableCell>{stats.mean.toFixed(1)}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">Median</TableCell>
          <TableCell>{stats.median}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">Std Dev</TableCell>
          <TableCell>{stats.std.toFixed(1)}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">p50</TableCell>
          <TableCell>{stats.p50}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">p95</TableCell>
          <TableCell>{stats.p95}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-muted-foreground">p99</TableCell>
          <TableCell>{stats.p99}</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  )
}

function generateTokenHistogram(tokenStats: any) {
  const max = tokenStats.max
  const bins = 10
  const binSize = Math.ceil(max / bins)
  const data = []
  
  // This is a simplified histogram - in reality you'd compute from actual data
  for (let i = 0; i < bins; i++) {
    const start = i * binSize
    const end = (i + 1) * binSize
    // Approximate using normal distribution around mean
    const center = (start + end) / 2
    const count = Math.max(1, Math.round(
      tokenStats.record_count * Math.exp(-0.5 * Math.pow((center - tokenStats.mean) / tokenStats.std, 2)) / (tokenStats.std * Math.sqrt(2 * Math.PI)) * binSize
    ))
    data.push({ range: `${start}-${end}`, count })
  }
  
  return data
}

async function exportDataset(datasetId: string) {
  try {
    const blob = await api.exportExperiment(datasetId, 'json')
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `dataset-${datasetId}-stats.json`
    a.click()
    URL.revokeObjectURL(url)
  } catch (err) {
    console.error('Export failed:', err)
  }
}