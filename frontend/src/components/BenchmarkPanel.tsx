import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Download, FlaskConical, Play, ShieldCheck } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { api, BenchmarkJob } from '../lib/api'
import { useBenchmark, useModels } from '../hooks/useData'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Progress } from './ui/progress'
import { Select } from './ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'

const policies = ['fixed', 'threshold', 'queue', 'ppo']
const format = (value: number | null) => value === null ? 'N/A' : value.toLocaleString(undefined, { maximumSignificantDigits: 5 })

export function BenchmarkPanel() {
  const [params, setParams] = useSearchParams()
  const id = params.get('benchmark')
  const { job, results, error: fetchError } = useBenchmark(id)
  const { models } = useModels()
  const [history, setHistory] = useState<BenchmarkJob[]>([])
  const [selected, setSelected] = useState(['fixed', 'threshold', 'queue'])
  const [workload, setWorkload] = useState('flash_crowd')
  const [seeds, setSeeds] = useState('11,22,33,44,55')
  const [duration, setDuration] = useState(600)
  const [deadline, setDeadline] = useState(30)
  const [target, setTarget] = useState(99)
  const [price, setPrice] = useState(0.01)
  const [baseline, setBaseline] = useState('fixed')
  const [model, setModel] = useState('')
  const [metric, setMetric] = useState('p95_latency_ms')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const running = submitting || job?.state === 'queued' || job?.state === 'running'

  useEffect(() => {
    let active = true
    api.listBenchmarks().then(data => { if (active) setHistory(data) })
      .catch(() => { if (active) setError('Could not load benchmark history') })
    return () => { active = false }
  }, [job?.state])

  const selectBenchmark = (value: string) => {
    const next = new URLSearchParams(params)
    if (value) next.set('benchmark', value)
    else next.delete('benchmark')
    setParams(next)
  }

  const start = async () => {
    setError(null)
    const parts = seeds.split(',').map(s => s.trim())
    if (parts.some(s => !/^\d+$/.test(s)) || new Set(parts.map(Number)).size !== parts.length || !selected.length) {
      setError('Select at least one policy and enter unique nonnegative integer seeds.')
      return
    }
    if (![duration, deadline, target, price].every(Number.isFinite)
      || duration < 1 || duration > 86400 || deadline < 1 || target < 0 || target > 100 || price < 0) {
      setError('Use a duration of 1–86400 seconds, a deadline of at least 1 second, a target of 0–100%, and a nonnegative replica price.')
      return
    }
    setSubmitting(true)
    try {
      const created = await api.createBenchmark({
        experiment: { episode_duration: duration, sla_threshold: deadline,
          slo_target: target / 100, cost_per_replica_second: price }, policies: selected,
        baseline_policy: baseline,
        workloads: [workload], seeds: parts.map(Number), num_requests: 5000,
        ...(selected.includes('ppo') ? { model_id: model } : {}),
      })
      selectBenchmark(created.id)
    } catch (err) {
      const message = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail
      setError(typeof message === 'string' ? message : 'Benchmark request failed. Check duration, deadline, target, price, seeds, and PPO checkpoint availability.')
    } finally {
      setSubmitting(false)
    }
  }

  const download = async (artifact: string) => {
    if (!id) return
    try {
      const blob = await api.exportBenchmark(id, artifact)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `benchmark-${id}-${artifact}.${artifact === 'report' ? 'md' : artifact === 'manifest' ? 'json' : 'csv'}`
      link.click()
      URL.revokeObjectURL(url)
    } catch {
      setError('Report download failed.')
    }
  }

  const rows = results?.aggregates.filter(row => row.metric === metric) ?? []
  const chart = rows.filter(row => row.mean !== null).map(row => ({
    policy: `${row.policy} (${row.workload})`, mean: row.mean,
    interval: row.ci_low !== null && row.ci_high !== null && row.mean !== null
      ? [row.mean - row.ci_low, row.ci_high - row.mean] : null,
  }))
  const terminal = job?.state === 'completed' || job?.state === 'failed'
  const evaluations = results?.cost_slo ?? []
  const frontier = evaluations.filter(row => row.mean_cost !== null && row.mean_attainment_lower !== null)
    .map(row => ({ ...row, cost: row.mean_cost, attainment: (row.mean_attainment_lower ?? 0) * 100,
      name: `${row.policy} (${row.workload})` }))
  const evaluatedTarget = (job?.config.experiment.slo_target ?? 0.99) * 100

  return <Card className="benchmark-panel">
    <CardHeader className="benchmark-panel-heading">
      <div className="benchmark-heading-icon"><FlaskConical size={21} strokeWidth={1.6} /></div>
      <div><p className="eyebrow mb-1">Controlled evaluation</p><CardTitle>Advanced benchmarking</CardTitle>
        <p className="mt-1.5 text-xs text-muted-foreground">Shared workloads. Independent seeds. A fairer comparison.</p></div>
      <span className="benchmark-confidence-note"><ShieldCheck size={13} /> 95% confidence intervals</span>
    </CardHeader>
    <CardContent className="space-y-6">
      <div className="benchmark-configuration space-y-5">
      <h3 className="configuration-heading"><span>01</span> Configure a new benchmark</h3>
      <div className="flex flex-wrap gap-2.5" role="group" aria-label="Benchmark policies">
        {policies.map(policy => <Label key={policy} className="policy-option" data-selected={selected.includes(policy)}>
          <input type="checkbox" checked={selected.includes(policy)} disabled={running}
            onChange={e => setSelected(e.target.checked ? [...selected, policy] : selected.filter(p => p !== policy))} />
          {policy.toUpperCase()}
        </Label>)}
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <div><Label htmlFor="benchmark-deadline">Response deadline (seconds)</Label>
          <Input id="benchmark-deadline" type="number" min={1} value={deadline} disabled={running}
            onChange={e => setDeadline(Number(e.target.value))} /></div>
        <div><Label htmlFor="benchmark-target">SLO target (% of arrivals)</Label>
          <Input id="benchmark-target" type="number" min={0} max={100} step={0.1} value={target} disabled={running}
            onChange={e => setTarget(Number(e.target.value))} /></div>
        <div><Label htmlFor="benchmark-price">USD per replica-second</Label>
          <Input id="benchmark-price" type="number" min={0} step="any" value={price} disabled={running}
            onChange={e => setPrice(Number(e.target.value))} /></div>
        <div><Label htmlFor="benchmark-baseline">Savings baseline</Label>
          <Select id="benchmark-baseline" value={baseline} disabled={running} onChange={e => setBaseline(e.target.value)}>
            {policies.map(policy => <option key={policy} value={policy}>{policy.toUpperCase()}</option>)}
          </Select></div>
      </div>
      <p className="text-[14px] leading-relaxed text-muted-foreground">Costs include startup, idle, active, and draining time. Select your baseline policy to calculate matched savings.</p>
      <div className="grid gap-4 md:grid-cols-3">
        <div><Label htmlFor="benchmark-workload">Workload</Label>
          <Select id="benchmark-workload" value={workload} disabled={running} onChange={e => setWorkload(e.target.value)}>
            {['steady', 'periodic', 'flash_crowd', 'mixed'].map(w => <option key={w}>{w}</option>)}
          </Select></div>
        <div><Label htmlFor="benchmark-seeds">Seeds (comma separated)</Label>
          <Input id="benchmark-seeds" value={seeds} disabled={running} onChange={e => setSeeds(e.target.value)} /></div>
        <div><Label htmlFor="benchmark-duration">Duration (simulation seconds)</Label>
          <Input id="benchmark-duration" type="number" min={1} max={86400} value={duration} disabled={running}
            onChange={e => setDuration(Number(e.target.value))} /></div>
      </div>
      {selected.includes('ppo') && <div>
        <Label htmlFor="benchmark-model">Existing trained PPO checkpoint</Label>
        <Select id="benchmark-model" value={model} onChange={e => setModel(e.target.value)} disabled={running}>
          <option value="">Select a registered model</option>
          {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </Select>
        <p className="text-sm text-muted-foreground">The server verifies that the checkpoint file exists. Training is never started by a benchmark.</p>
      </div>}
      <Button onClick={start} disabled={running || (selected.includes('ppo') && !model)}>
        <Play size={13} />
        {running ? 'Benchmark running…' : 'Run multi-seed benchmark'}
      </Button>
      </div>
      {(error || fetchError) && <p role="alert" className="text-destructive">{error || fetchError}</p>}
      <div className="benchmark-history"><div><h3 className="configuration-heading"><span>02</span> Explore your results</h3><p className="mt-1.5 text-[14px] text-muted-foreground">Pick up where you left off.</p></div><div className="min-w-0 flex-1 sm:max-w-sm"><Label htmlFor="benchmark-history" className="sr-only">Saved benchmarks</Label>
        <Select id="benchmark-history" value={id || ''} onChange={e => selectBenchmark(e.target.value)}>
          <option value="">Select a benchmark</option>
          {history.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.state}</option>)}
        </Select></div></div>
      {id && !job && !fetchError && <p role="status">Loading benchmark…</p>}
      {job && <div className="benchmark-progress" aria-live="polite">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3"><span className={`status-pill status-${job.state}`}>{job.state}</span><p className="text-xs text-muted-foreground">{Math.round(job.progress * 100)}%{results && ` · ${results.successful_runs} successful / ${results.failed_runs} failed runs`}</p></div>
        <Progress value={job.progress * 100} aria-label="Benchmark completion" />
        {job.error && <p role="alert" className="text-destructive">{job.error}</p>}
      </div>}
      {results && <>
        <div className="benchmark-results space-y-4">
          <h3 className="section-heading">Cost and SLO evaluation</h3>
          {job && <div className="run-settings-grid"><div><span>Response deadline</span><strong>{job.config.experiment.sla_threshold ?? 30}<small> sec</small></strong></div><div><span>SLO target</span><strong>{format(evaluatedTarget)}<small>%</small></strong></div><div><span>Episode duration</span><strong>{job.config.experiment.episode_duration}<small> sec</small></strong></div><div><span>Replica rate</span><strong>${job.config.experiment.cost_per_replica_second ?? 0.01}<small> / sec</small></strong></div></div>}
          <p className="text-[14px] leading-relaxed text-muted-foreground">Met requires every seed to reach the target. Violated means at least one seed cannot reach it. Pending outcomes may make a result inconclusive.</p>
          {frontier.length > 0 && <div className="benchmark-chart-surface" aria-label="Mean operating cost versus lower SLO attainment bound">
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 15, right: 30, bottom: 25, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="cost" name="Mean cost" unit=" USD" label={{ value: 'Mean simulated cost (USD)', position: 'bottom' }} />
                <YAxis type="number" dataKey="attainment" name="Mean lower attainment bound" unit="%" domain={[0, 100]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                  const point = payload?.[0]?.payload as (typeof frontier)[number] | undefined
                  return active && point ? <div className="rounded-lg border bg-card p-3 text-xs shadow-lg">
                    <p className="font-semibold">{point.name}</p>
                    <p>Mean cost: USD {format(point.cost)}</p>
                    <p>Mean lower attainment: {format(point.attainment)}%</p>
                    <p>SLO: {point.slo_status}</p>
                  </div> : null
                }} />
                <ReferenceLine y={evaluatedTarget} stroke="#64748b" strokeDasharray="4 4" />
                <Scatter name="Policies" data={frontier}>
                  {frontier.map(row => <Cell key={row.name} fill={row.slo_status === 'met' ? '#2da889' : row.slo_status === 'violated' ? '#dc7070' : '#cb9a48'} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>}
          <Table><TableHeader><TableRow>
            {['Policy / workload', 'Mean cost (USD)', 'SLO status', 'Met / violated / inconclusive', 'Mean attainment bounds', 'Mean overdue / pending', 'Savings vs baseline', 'Matched pairs'].map(h => <TableHead key={h}>{h}</TableHead>)}
          </TableRow></TableHeader><TableBody>{evaluations.map(row => <TableRow key={`${row.policy}-${row.workload}`}>
            <TableCell>{row.policy} / {row.workload}</TableCell>
            <TableCell className="font-semibold">{format(row.mean_cost)}</TableCell><TableCell><span className={`status-pill status-${row.slo_status}`}>{row.slo_status}</span></TableCell>
            <TableCell>{row.slo_met_runs} / {row.slo_violated_runs} / {row.slo_inconclusive_runs} ({row.completed_runs}/{row.expected_runs} evaluated{row.failed_runs > 0 && `, ${row.failed_runs} failed`})</TableCell>
            <TableCell>{row.mean_attainment_lower === null ? 'N/A' : `${format(row.mean_attainment_lower * 100)}–${format((row.mean_attainment_upper ?? 0) * 100)}%`}</TableCell>
            <TableCell>{format(row.mean_overdue_requests)} / {format(row.mean_pending_requests)}</TableCell>
            <TableCell>{row.mean_cost_savings_percent === null ? 'N/A' : `${format(row.mean_cost_savings_percent)}%`} vs {row.baseline_policy}</TableCell>
            <TableCell>{row.matched_cost_pairs}</TableCell>
          </TableRow>)}</TableBody></Table>
          <details className="benchmark-methodology"><summary>How to interpret cost and SLO results</summary><p>The chart shows means; SLO status checks each run separately. Pending requests have unexpired deadlines. Overdue unfinished requests count as violations. Savings alone do not imply SLO compliance. Older benchmarks use startup-only accounting and must be rerun for this evaluation. Seeds: {job?.config.seeds.join(', ')}.</p></details>
        </div>
        <div><Label htmlFor="benchmark-metric">Comparison metric</Label>
          <Select id="benchmark-metric" value={metric} onChange={e => setMetric(e.target.value)}>
            {[...new Set(results.aggregates.map(row => row.metric))].map(m => <option key={m}>{m}</option>)}
          </Select></div>
        {chart.length > 0 && <div aria-label="Run means and 95 percent confidence intervals">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chart}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="policy" />
              <YAxis domain={['auto', 'auto']} /><Tooltip />
              <Bar dataKey="mean" fill="#287d68" radius={[5, 5, 0, 0]} maxBarSize={90}><ErrorBar dataKey="interval" width={6} strokeWidth={1.5} /></Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>}
        <Table><TableHeader><TableRow>
          {['Policy', 'Workload', 'Valid n', 'Mean', 'Sample SD', '95% CI', 'Failed / missing'].map(h => <TableHead key={h}>{h}</TableHead>)}
        </TableRow></TableHeader><TableBody>{rows.map(row => <TableRow key={`${row.policy}-${row.workload}`}>
          <TableCell>{row.policy}</TableCell><TableCell>{row.workload}</TableCell><TableCell>{row.n}</TableCell>
          <TableCell>{format(row.mean)}</TableCell><TableCell>{format(row.std)}</TableCell>
          <TableCell>{row.ci_low === null ? 'N/A' : `[${format(row.ci_low)}, ${format(row.ci_high)}]`}</TableCell>
          <TableCell>{row.failed} / {row.missing}</TableCell>
        </TableRow>)}</TableBody></Table>
        <p className="text-sm text-muted-foreground">Intervals describe uncertainty in the mean across runs. N/A means insufficient independent observations. Partial results may change while running.</p>
        {results.failed_runs > 0 && <details><summary>Failed runs</summary><ul>
          {results.result.runs.filter(r => r.status === 'failed').map(r => <li key={`${r.policy}-${r.workload}-${r.seed}`}>{r.policy}/{r.workload}/{r.seed}: {r.error}</li>)}
        </ul></details>}
        <details className="benchmark-methodology"><summary>Methodology and limitations</summary><ul>
          {results.result.warnings.map(w => <li key={w}>{w}</li>)}
        </ul></details>
        {terminal && <div className="flex flex-wrap gap-2">
          {['runs', 'aggregates', 'report', 'manifest'].map(a => <Button variant="outline" key={a} onClick={() => download(a)}><Download size={13} />{a === 'runs' ? 'Raw runs' : a === 'aggregates' ? 'Summary CSV' : a === 'report' ? 'Research report' : 'Manifest'}</Button>)}
        </div>}
      </>}
    </CardContent>
  </Card>
}
