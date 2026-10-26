import { useEffect, useState } from 'react'
import { isAxiosError } from 'axios'
import { Activity, ArrowRight, Brain, CheckCircle2, Download, Loader2, Pause, Play, RefreshCw, ShieldCheck } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Label } from '../components/ui/label'
import { Select } from '../components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table'
import { studyApi, type StudyData, type StudyPolicy, type StudyProfile, type StudyReplay, type StudyStats } from '../lib/study'

const names: Record<StudyPolicy, string> = { fixed: 'Fixed', threshold: 'Threshold', queue: 'Queue', ppo: 'PPO' }
const colors: Record<StudyPolicy, string> = { fixed: '#8b9a95', threshold: '#bf925e', queue: '#6b8da6', ppo: '#288c70' }
const money = (n: number) => '$' + n.toFixed(3)
const percent = (n: number) => (n * 100).toFixed(2) + '%'
const seconds = (ms: number) => (ms / 1000).toFixed(3) + ' s'
const actions = ['Scale down', 'Hold', 'Scale up']
const exportBase = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '') + '/rl-study/export/'
const tooltipStyle = { background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 10, color: 'hsl(var(--foreground))', fontSize: 14 }

function message(error: unknown) {
  return isAxiosError(error) && typeof error.response?.data?.detail === 'string'
    ? error.response.data.detail : error instanceof Error ? error.message : 'The study could not be loaded.'
}

export function RLStudy() {
  const [data, setData] = useState<StudyData | null>(null)
  const [error, setError] = useState('')
  const [profile, setProfile] = useState<StudyProfile>('bursty')
  const [seed, setSeed] = useState(1001)
  const [replay, setReplay] = useState<StudyReplay | null>(null)
  const [running, setRunning] = useState(false)
  const [replayError, setReplayError] = useState('')
  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)

  const load = () => {
    setError('')
    studyApi.get().then(setData).catch(err => setError(message(err)))
  }
  useEffect(() => {
    let active = true
    studyApi.get().then(value => { if (active) setData(value) }).catch(err => { if (active) setError(message(err)) })
    return () => { active = false }
  }, [])
  const lastStep = Math.max(0, (replay?.runs[0]?.timeline.length ?? 1) - 1)
  useEffect(() => {
    if (!playing || cursor >= lastStep) return
    const timer = setTimeout(() => setCursor(value => Math.min(value + 1, lastStep)), 350)
    return () => clearTimeout(timer)
  }, [playing, cursor, lastStep])

  const runReplay = async () => {
    setRunning(true)
    setReplayError('')
    setPlaying(false)
    try {
      const result = await studyApi.replay(profile, seed)
      setReplay(result)
      setCursor(0)
      setPlaying(true)
    } catch (err) { setReplayError(message(err)) }
    finally { setRunning(false) }
  }

  if (error) return <div className="p-6 space-y-4"><h1 className="page-heading">RL Study</h1><Card><CardHeader><CardTitle>Study unavailable</CardTitle><CardDescription>{error}</CardDescription></CardHeader><CardContent><Button onClick={load}><RefreshCw size={16} /> Retry</Button></CardContent></Card></div>
  if (!data) return <div className="p-8 flex items-center gap-3" role="status"><Loader2 className="animate-spin" size={20} /> Loading trained-agent evidence…</div>

  const rows = data.summaries.filter(row => row.profile === profile)
  const ppo = rows.find(row => row.policy === 'ppo')!
  const fixed = rows.find(row => row.policy === 'fixed')!
  const ppoMet = data.summaries.filter(row => row.policy === 'ppo').reduce((sum, row) => sum + row.slo_met_runs, 0)
  const ppoTotal = data.summaries.filter(row => row.policy === 'ppo').reduce((sum, row) => sum + row.run_count, 0)
  const chartRows = rows.map(row => {
    const field = (stats: StudyStats, factor = 1) => ({
      mean: stats.mean * factor,
      error: [(stats.mean - (stats.ci_low ?? stats.mean)) * factor, ((stats.ci_high ?? stats.mean) - stats.mean) * factor],
    })
    return { policy: row.policy, name: names[row.policy], cost: field(row.total_cost), attainment: field(row.slo_attainment_lower_bound, 100), latency: field(row.p95_latency_ms, .001) }
  })
  const bars = [
    { key: 'cost', title: 'Operating cost', description: 'Billable starting, active, and draining time.', unit: 'USD / episode', target: undefined },
    { key: 'attainment', title: 'Deadline attainment', description: 'Confirmed on-time requests, including unfinished work in the denominator.', unit: '% of arrivals', target: data.manifest.config.slo_target * 100 },
    { key: 'latency', title: 'Response latency', description: 'Mean of each run’s p95 for completed requests.', unit: 'Seconds', target: data.manifest.config.sla_threshold },
  ]
  const ppoReplay = replay?.runs.find(row => row.policy === 'ppo')
  const fixedReplay = replay?.runs.find(row => row.policy === 'fixed')
  const frame = ppoReplay?.timeline[cursor]
  const frames = ppoReplay?.timeline.map((point, index) => ({
    time: point.time, ppoReplicas: point.active, fixedReplicas: fixedReplay?.timeline[index]?.active ?? 0,
    ppoQueue: point.queue, fixedQueue: fixedReplay?.timeline[index]?.queue ?? 0,
  })) ?? []
  const visibleFrames = frames.slice(0, cursor + 1)
  const selectedRuns = data.runs.filter(row => row.profile === profile)

  return <div className="overview-page space-y-7">
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div><p className="eyebrow mb-2">Trained agent · recorded evidence</p><h1 className="page-heading">RL Study</h1>
        <p className="page-description">Follow the learning. Compare the tradeoffs. Replay the trained policy.</p></div>
      <div className="flex flex-wrap gap-2">
        <a href={exportBase + 'report'} target="_blank" rel="noreferrer" className="text-link h-11 px-2">Two-page report <ArrowRight size={15} /></a>
        <a href={exportBase + 'runs'} className="inline-flex h-11 items-center gap-2 rounded-lg border border-border bg-card px-4 text-sm font-semibold"><Download size={16} /> Run data</a>
      </div>
    </div>

    <div className="surface-card flex flex-wrap items-center justify-between gap-4 p-5">
      <div className="flex items-center gap-3"><span className="stat-icon stat-icon-green"><ShieldCheck size={20} /></span>
        <div><p className="font-semibold">Saved PPO checkpoint verified</p><p className="text-sm text-muted-foreground">{data.checkpoint.selected_step.toLocaleString()} environment steps · entropy {data.manifest.policy.ent_coef} · deterministic inference</p></div></div>
      <a className="text-link" href={exportBase + 'checkpoint'}><Download size={16} /> Trained agent</a>
    </div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {[
        { label: 'Held-out comparisons', value: data.runs.length, detail: '4 policies × 3 profiles × 5 seeds' },
        { label: 'PPO runs meeting the SLO', value: ppoMet + '/' + ppoTotal, detail: '95% attainment within 2 seconds' },
        { label: 'Selected training budget', value: data.manifest.training_steps.toLocaleString(), detail: 'Two candidates; validation-only selection' },
        { label: 'Inference environment', value: 'CPU simulation', detail: 'Synthetic arrivals and token sizes' },
      ].map(stat => <div className="surface-card p-5" key={stat.label}><p className="text-sm text-muted-foreground">{stat.label}</p><p className="my-2 text-2xl font-semibold tracking-tight">{stat.value}</p><p className="text-sm text-muted-foreground">{stat.detail}</p></div>)}
    </div>

    <Card>
      <CardHeader><CardTitle>What did PPO learn?</CardTitle><CardDescription>Actual stochastic training returns (rolling 20 episodes) and deterministic validation returns. These are different measurements; higher return is better.</CardDescription></CardHeader>
      <CardContent><div className="h-[280px] min-w-0" aria-label="Measured PPO learning history">
        <ResponsiveContainer width="100%" height="100%"><LineChart data={data.learning} margin={{ top: 10, right: 15, bottom: 10, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="timesteps" type="number" domain={[0, data.manifest.training_steps]} tickFormatter={value => (Number(value) / 1000) + 'k'} tick={{ fontSize: 13 }} /><YAxis tick={{ fontSize: 13 }} width={45} />
          <Tooltip contentStyle={tooltipStyle} labelFormatter={value => Number(value).toLocaleString() + ' steps'} formatter={value => Number(value).toFixed(3)} /><Legend />
          <ReferenceLine x={data.manifest.selected_step} stroke="#bf925e" strokeDasharray="4 4" />
          <Line name="Training return · rolling 20" type="monotone" dataKey="training_return" stroke="#849b92" dot={false} connectNulls isAnimationActive={false} />
          <Line name="Validation return · deterministic" type="linear" dataKey="validation_return" stroke="#288c70" strokeWidth={2.5} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
        </LineChart></ResponsiveContainer>
      </div><p className="mt-3 text-sm text-muted-foreground">Validation improves over early trained checkpoints, but stays below the untrained control. The dotted line marks the selected checkpoint.</p></CardContent>
    </Card>

    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4"><div><h2 className="section-heading">Cost and service quality, together</h2><p className="mt-1 text-sm text-muted-foreground">Shared test traces. Five request seeds per profile. Error bars show 95% Student t intervals.</p></div>
        <div className="w-full sm:w-52"><Label htmlFor="study-profile">Traffic profile</Label><Select id="study-profile" value={profile} disabled={running} onValueChange={value => { setProfile(value as StudyProfile); setReplay(null); setPlaying(false) }}>
          {data.profiles.map(value => <option key={value} value={value}>{value[0].toUpperCase() + value.slice(1)}{value === 'shifted' ? ' · unseen pattern' : ''}</option>)}</Select></div></div>
      <div className="grid gap-5 xl:grid-cols-3">
        {bars.map(metric => <Card key={metric.key} className="min-w-0">
          <CardHeader><CardTitle>{metric.title}</CardTitle><CardDescription>{metric.description}</CardDescription><p className="pt-1 text-sm text-muted-foreground">{metric.unit}</p></CardHeader>
          <CardContent><div className="h-[250px]" aria-label={metric.title + ' by policy'}>
            <ResponsiveContainer width="100%" height="100%"><BarChart data={chartRows} margin={{ top: 15, right: 8, left: -18, bottom: 5 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 13 }} tickLine={false} interval={0} /><YAxis tick={{ fontSize: 13 }} domain={metric.key === 'attainment' ? [0, 110] : [0, 'auto']} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'hsl(var(--muted) / .6)' }} formatter={value => [metric.key === 'cost' ? money(Number(value)) : Number(value).toFixed(2) + (metric.key === 'attainment' ? '%' : ' s'), metric.title]} />
              {metric.target !== undefined && <ReferenceLine y={metric.target} stroke="#bf765e" strokeDasharray="4 4" />}
              <Bar dataKey={metric.key + '.mean'} maxBarSize={48} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                {chartRows.map(row => <Cell key={row.policy} fill={colors[row.policy]} />)}
                <ErrorBar dataKey={metric.key + '.error'} width={4} stroke="hsl(var(--foreground))" strokeWidth={1.2} />
              </Bar>
            </BarChart></ResponsiveContainer>
          </div></CardContent>
        </Card>)}
      </div>
      <div className="rounded-xl border border-primary/20 bg-primary/5 px-5 py-4 text-sm leading-relaxed">
        <strong>Read the tradeoff:</strong> on {profile} traffic, PPO averages {percent(ppo.slo_attainment_lower_bound.mean)} on time at {money(ppo.total_cost.mean)}; fixed averages {percent(fixed.slo_attainment_lower_bound.mean)} at {money(fixed.total_cost.mean)}. Lower cost alone does not establish better service.
      </div>
    </section>

    <Card>
      <CardHeader><div className="flex items-center gap-2"><Activity size={19} className="text-primary" /><CardTitle>Run the trained agent</CardTitle></div>
        <CardDescription>Load the saved PPO checkpoint and replay one reserved test trace against fixed capacity. The simulation runs now; playback then reveals the recorded decision points.</CardDescription></CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-36"><Label htmlFor="replay-seed">Request seed</Label><Select id="replay-seed" value={seed} disabled={running} onValueChange={value => { setSeed(Number(value)); setReplay(null); setPlaying(false) }}>{data.seeds.map(value => <option key={value} value={value}>{value}</option>)}</Select></div>
          <Button onClick={runReplay} disabled={running}>{running ? <Loader2 size={17} className="animate-spin" /> : <Brain size={17} />}{running ? 'Running real inference…' : 'Run PPO replay'}</Button>
          <span className="pb-2 text-sm text-muted-foreground">{profile} · {data.manifest.config.episode_duration}s simulated · no retraining</span>
        </div>
        {replayError && <p role="alert" className="text-sm text-destructive">{replayError}</p>}
        {replay && <>
          <div role="status" className="flex flex-wrap items-center gap-2 text-sm text-primary"><CheckCircle2 size={17} /> Fresh inference completed · {replay.request_count.toLocaleString()} identical requests per policy · {replay.runs.every(row => row.matches_recorded) ? 'Matches saved evaluation' : 'Metrics differ from the saved evaluation'}</div>
          <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>Policy</TableHead><TableHead>Cost</TableHead><TableHead>On time</TableHead><TableHead>p95 latency</TableHead><TableHead>SLO</TableHead></TableRow></TableHeader>
            <TableBody>{replay.runs.map(row => <TableRow key={row.policy}><TableCell className="font-semibold">{names[row.policy]}</TableCell><TableCell>{money(row.metrics.total_cost)}</TableCell><TableCell>{percent(row.metrics.slo_attainment_lower_bound)}</TableCell><TableCell>{seconds(row.metrics.p95_latency_ms)}</TableCell><TableCell>{row.metrics.slo_attainment_lower_bound >= data.manifest.config.slo_target ? 'Met' : 'Not met'}</TableCell></TableRow>)}</TableBody></Table></div>
          <div className="rounded-xl border border-border bg-muted/30 p-4 space-y-4">
            <div className="flex flex-wrap items-center gap-3"><Button variant="outline" size="sm" onClick={() => { if (cursor >= lastStep) { setCursor(0); setPlaying(true) } else { setPlaying(value => !value) } }}>{playing && cursor < lastStep ? <Pause size={15} /> : <Play size={15} />}{playing && cursor < lastStep ? 'Pause playback' : 'Play decisions'}</Button><span className="text-sm font-medium">Simulation time: {frame?.time ?? 0}s / {data.manifest.config.episode_duration}s</span></div>
            <Label htmlFor="study-playback" className="sr-only">Decision playback position</Label><input id="study-playback" type="range" min={0} max={lastStep} value={cursor} onChange={event => { setCursor(Number(event.target.value)); setPlaying(false) }} className="w-full accent-primary" />
            <div className="grid gap-3 text-sm sm:grid-cols-4">{[
              ['PPO active replicas', frame?.active ?? 0], ['Replicas starting', frame?.starting ?? 0],
              ['Requests waiting', frame?.queue ?? 0], ['Policy request', actions[frame?.action ?? 1]],
            ].map(([label, value]) => <div key={label} className="rounded-lg border border-border bg-card p-3"><p className="text-muted-foreground">{label}</p><p className="mt-1 text-lg font-semibold">{value}</p></div>)}</div>
          </div>
          <div className="grid gap-5 lg:grid-cols-2">{[
            { title: 'Active serving replicas', ppo: 'ppoReplicas', fixed: 'fixedReplicas' },
            { title: 'Waiting requests', ppo: 'ppoQueue', fixed: 'fixedQueue' },
          ].map(metric => <div className="min-w-0" key={metric.title}><p className="mb-3 font-semibold">{metric.title}</p><div className="h-[240px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={visibleFrames} margin={{ right: 12, left: -18, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="time" type="number" domain={[0, data.manifest.config.episode_duration]} tickFormatter={value => value + 's'} tick={{ fontSize: 13 }} /><YAxis allowDecimals={false} tick={{ fontSize: 13 }} /><Tooltip contentStyle={tooltipStyle} labelFormatter={value => 'Simulation time: ' + value + 's'} /><Legend />
            <Line name="PPO" type="stepAfter" dataKey={metric.ppo} stroke={colors.ppo} strokeWidth={2.5} dot={false} isAnimationActive={false} /><Line name="Fixed" type="stepAfter" dataKey={metric.fixed} stroke={colors.fixed} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart></ResponsiveContainer></div></div>)}</div>
          <p className="text-sm text-muted-foreground">Actions are policy requests; cooldown, startup, and replica limits determine when changes take effect. Points show state immediately before each decision. Playback ends at the last decision; final metrics include the full episode.</p>
          <details className="text-sm"><summary className="cursor-pointer font-medium">Trace and checkpoint identity</summary><p className="mt-2 break-all text-muted-foreground">Trace SHA256: {replay.trace_hash}<br />Checkpoint SHA256: {replay.checkpoint_sha256}</p></details>
        </>}
      </CardContent>
    </Card>

    <Card><CardHeader><CardTitle>Inspect the individual runs</CardTitle><CardDescription>Every value below comes from a saved test run for the selected profile.</CardDescription></CardHeader><CardContent>
      <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>Seed</TableHead><TableHead>Policy</TableHead><TableHead>Cost</TableHead><TableHead>On time</TableHead><TableHead>p95 latency</TableHead><TableHead>Overdue</TableHead><TableHead>Pending</TableHead></TableRow></TableHeader>
        <TableBody>{selectedRuns.map(row => <TableRow key={row.seed + row.policy}><TableCell>{row.seed}</TableCell><TableCell className={row.policy === 'ppo' ? 'font-semibold text-primary' : ''}>{names[row.policy]}</TableCell><TableCell>{money(row.metrics.total_cost)}</TableCell><TableCell>{percent(row.metrics.slo_attainment_lower_bound)}</TableCell><TableCell>{seconds(row.metrics.p95_latency_ms)}</TableCell><TableCell>{row.metrics.overdue_requests}</TableCell><TableCell>{row.metrics.pending_requests}</TableCell></TableRow>)}</TableBody>
      </Table></div>
    </CardContent></Card>

    <details className="surface-card p-5 text-sm"><summary className="cursor-pointer text-base font-semibold">Experiment setup and honest limits</summary>
      <div className="mt-4 grid gap-5 md:grid-cols-2"><div className="space-y-2"><p><strong>State:</strong> 12 raw queue, token, replica, latency, deadline, and cooldown features. Fixed input scaling travels with the checkpoint.</p><p><strong>Actions:</strong> scale down, hold, scale up.</p><p><strong>Reward:</strong> penalties for interval cost (0.1), new deadline misses (1.0), queue pressure (0.01), and scaling churn (0.01).</p><p><strong>Profile rates:</strong> {data.manifest.rates[profile].join(' → ')} requests/s, with seeded variation.</p></div>
        <div className="space-y-2"><p><strong>Split:</strong> training 100–199; validation 501–503; testing 1001–1005. Shifted traffic is test-only.</p><p><strong>Capacity:</strong> 1–8 replicas, concurrency 2, startup 5s, cooldown 10s.</p><p>Results are synthetic CPU simulation. They do not demonstrate real GPU savings, production-trace training, or Kubernetes HPA equivalence. Test error bars cover workload variation for one trained agent.</p><a href={exportBase + 'manifest'} className="text-link"><Download size={15} /> Download training configuration</a></div></div>
      <p className="mt-4 break-all text-muted-foreground">Checkpoint SHA256: {data.checkpoint.sha256}</p>
    </details>
  </div>
}
