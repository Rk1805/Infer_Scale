import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Activity, ArrowDownRight, ArrowRight, ArrowUpRight, Boxes, Brain, Check, ChevronRight, CircleDot, Database, FlaskConical, GitCompare, Layers3, Plus, RefreshCw } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatDistanceToNow } from 'date-fns'
import { useBenchmark, useDatasets, useExperiments, useModels } from '../hooks/useData'
import { api, BenchmarkJob } from '../lib/api'
import { cn } from '../lib/utils'
import { Button } from '../components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table'

const policyNames: Record<string, string> = { fixed: 'Fixed capacity', threshold: 'Threshold', queue: 'Queue-based', ppo: 'PPO' }
const policyColors: Record<string, string> = { fixed: '#287d68', threshold: '#85a190', queue: '#667c80', ppo: '#b28b57' }
const money = (value: number) => `$${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}`

function relativeTime(value: string | null) {
  if (!value) return 'Not started'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : formatDistanceToNow(date, { addSuffix: true })
}

export function Overview() {
  const navigate = useNavigate()
  const { experiments, loading: experimentsLoading, error: experimentsError, refetch } = useExperiments()
  const { models, loading: modelsLoading } = useModels()
  const { datasets, loading: datasetsLoading } = useDatasets()
  const [benchmarks, setBenchmarks] = useState<BenchmarkJob[]>([])
  const [benchmarksLoading, setBenchmarksLoading] = useState(true)
  const [benchmarksError, setBenchmarksError] = useState<string | null>(null)
  const latest = benchmarks.find(item => item.state === 'completed')
  const { results, error: resultsError } = useBenchmark(latest?.id ?? null)
  const workload = latest?.config.workloads[0]
  const comparisons = results?.cost_slo.filter(row => row.workload === workload && row.mean_cost !== null) ?? []
  const chart = comparisons.map(row => ({ name: policyNames[row.policy] ?? row.policy, cost: row.mean_cost, policy: row.policy }))
  const completeExperiments = experiments.filter(item => item.state === 'completed').length
  const activeExperiments = experiments.filter(item => item.state === 'running').length

  useEffect(() => {
    let active = true
    api.listBenchmarks().then(data => { if (active) setBenchmarks(data) })
      .catch(() => { if (active) setBenchmarksError('Benchmark history could not be loaded.') })
      .finally(() => { if (active) setBenchmarksLoading(false) })
    return () => { active = false }
  }, [])

  const stats = [
    { label: 'Total experiments', value: experimentsLoading ? '—' : experiments.length.toLocaleString(), detail: `${completeExperiments} completed`, icon: FlaskConical, color: 'violet', suffix: activeExperiments ? `${activeExperiments} running` : null },
    { label: 'Saved benchmarks', value: benchmarksLoading ? '—' : benchmarks.length.toLocaleString(), detail: 'Reproducible policy comparisons', icon: GitCompare, color: 'blue', suffix: null },
    { label: 'Registered models', value: modelsLoading ? '—' : models.length.toLocaleString(), detail: 'PPO model registry', icon: Brain, color: 'amber', suffix: null },
    { label: 'Available datasets', value: datasetsLoading ? '—' : datasets.length.toLocaleString(), detail: 'Workload sources', icon: Database, color: 'green', suffix: null },
  ]

  return <div className="overview-page space-y-7">
    <div className="overview-heading">
      <div><p className="eyebrow mb-2.5">Your research, at a glance</p><h1 className="page-heading">Research overview</h1>
        <p className="page-description">A clearer view of your inference experiments, costs, and performance.</p></div>
      <Button onClick={() => navigate('/experiment-studio')}><Plus size={15} /> New experiment</Button>
    </div>
    {(experimentsError || benchmarksError || resultsError) && <div role="alert" className="rounded-lg border border-destructive/20 bg-destructive/5 px-4 py-3 text-xs text-destructive">{experimentsError || benchmarksError || resultsError}</div>}
    <div className="overview-stats">
      {stats.map(stat => <div className="surface-card stat-card" key={stat.label}>
        <div className="stat-topline"><span>{stat.label}</span><span className={cn('stat-icon', `stat-icon-${stat.color}`)}><stat.icon size={16} strokeWidth={1.7} /></span></div>
        <div className="stat-value">{stat.value}</div>
        <div className="stat-caption"><span>{stat.detail}</span>{stat.suffix && <span className="stat-live">{stat.suffix}</span>}</div>
      </div>)}
    </div>
    <div className="overview-primary-grid">
      <section className="surface-card overview-cost-card">
        <div className="overview-card-heading">
          <div><div className="flex items-center gap-2"><span className="section-icon"><GitCompare size={16} /></span><h2 className="section-heading">Policy cost comparison</h2></div>
            <p className="mt-2 text-xs text-muted-foreground">{latest ? `Latest completed benchmark · ${workload?.replace(/_/g, ' ')}` : 'Compare the operating cost of your scaling policies.'}</p></div>
          {results && <span className="chart-run-label"><span /> {results.successful_runs} simulation runs</span>}
        </div>
        {chart.length > 0 ? <>
          <div className="overview-chart" aria-label="Mean simulated operating cost by policy">
            <span className="chart-unit-label">MEAN COST · USD</span>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chart} margin={{ top: 30, right: 15, left: -20, bottom: 0 }} barSize={48}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} dy={10} />
                <YAxis axisLine={false} tickLine={false} tickFormatter={value => money(Number(value))} tickCount={5} />
                <Tooltip formatter={value => [money(Number(value)), 'Mean operating cost']} cursor={{ fill: 'hsl(var(--muted) / .5)' }} />
                <Bar dataKey="cost" radius={[5, 5, 0, 0]}>
                  {chart.map(row => <Cell key={row.policy} fill={policyColors[row.policy]} />)}
                  <LabelList dataKey="cost" position="top" formatter={(value: number) => money(value)} className="cost-bar-label" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overview-chart-footer"><span><CircleDot size={13} /> Costs are simulated. Check SLO outcomes alongside savings.</span>
            <Link className="text-link" to={`/policy-comparison?benchmark=${latest?.id}`}>Explore results <ArrowUpRight size={14} /></Link></div>
        </> : <div className="chart-empty-state">
          <div className="empty-chart-bars" aria-hidden="true"><span /><span /><span /><span /></div>
          <h3>{benchmarksLoading || (latest && !results && !resultsError) ? 'Loading benchmark results' : 'Your next insight starts here'}</h3>
          <p>Run a benchmark to see real policy costs and response-time outcomes.</p>
          <Link className="text-link" to="/policy-comparison">Compare policies <ArrowRight size={14} /></Link>
        </div>}
      </section>
      <div className="overview-right-column">
        <section className="experiment-launch-card">
          <div className="launch-card-kicker"><span /><span>THE EXPERIMENT STUDIO</span></div>
          <div className="replica-illustration" aria-hidden="true"><div className="replica-line" />
            {[0, 1, 2].map(index => <div className="replica-node" key={index}><Boxes size={23} strokeWidth={1.3} /><span /><span /><span /></div>)}
          </div>
          <h2>Make your next<br />scaling decision smarter.</h2>
          <p>Choose a workload. Set your targets.<br />See how your policies perform.</p>
          <Link to="/experiment-studio" className="launch-card-link">Create an experiment <ArrowUpRight size={16} /></Link>
        </section>
        <div className="surface-card environment-card"><span className="environment-icon"><Layers3 size={19} /></span>
          <div><strong>CPU simulation environment</strong><p>Explore policies without a GPU.</p></div><span className="environment-chip">LOCAL</span></div>
      </div>
    </div>
    <div className="overview-bottom-grid">
      <section className="surface-card overflow-hidden">
        <div className="overview-card-heading">
          <div className="flex items-center gap-2"><span className="section-icon"><Activity size={16} /></span><h2 className="section-heading">Recent experiments</h2></div>
          <div className="flex items-center gap-3"><button className="icon-control !h-7 !w-7" aria-label="Refresh experiments" onClick={refetch}><RefreshCw size={14} className={experimentsLoading ? 'animate-spin' : ''} /></button><Link to="/experiment-history" className="text-link">View all <ArrowRight size={13} /></Link></div>
        </div>
        {experiments.length > 0 ? <Table>
          <TableHeader><TableRow><TableHead>Experiment</TableHead><TableHead>Policy</TableHead><TableHead>Status</TableHead><TableHead>Started</TableHead><TableHead><span className="sr-only">Open</span></TableHead></TableRow></TableHeader>
          <TableBody>{experiments.slice(0, 5).map(experiment => <TableRow key={experiment.id}>
            <TableCell><Link className="experiment-id" to={`/simulation/${experiment.id}`}><FlaskConical size={14} />{experiment.id}</Link><span className="experiment-workload">{experiment.config.workload_type.replace(/_/g, ' ')}</span></TableCell>
            <TableCell><span className="policy-table-label">{policyNames[experiment.config.policy] ?? experiment.config.policy}</span></TableCell>
            <TableCell><span className={cn('status-pill', `status-${experiment.state}`)}>{experiment.state}</span></TableCell>
            <TableCell className="text-[14px] text-muted-foreground">{relativeTime(experiment.started_at)}</TableCell>
            <TableCell><Link to={`/simulation/${experiment.id}`} aria-label={`Open experiment ${experiment.id}`} className="icon-control !h-6 !w-6"><ChevronRight size={15} /></Link></TableCell>
          </TableRow>)}</TableBody>
        </Table> : <div className="py-14 text-center"><FlaskConical size={24} className="mx-auto mb-3 text-muted-foreground/50" /><p className="text-sm font-medium">No experiments yet</p><p className="mt-1 text-xs text-muted-foreground">Create your first experiment to start exploring.</p></div>}
      </section>
      <section className="surface-card research-workflow">
        <div className="overview-card-heading"><h2 className="section-heading">From question to insight</h2><ArrowDownRight size={18} className="text-muted-foreground" /></div>
        {[
          { step: '01', title: 'Choose your workload', detail: 'Understand the traffic you want to serve.', href: '/dataset-explorer', icon: Database },
          { step: '02', title: 'Put policies to the test', detail: 'Compare cost and response-time targets.', href: '/policy-comparison', icon: GitCompare },
          { step: '03', title: 'Keep the evidence', detail: 'Revisit runs and export your findings.', href: '/experiment-history', icon: Check },
        ].map(item => <Link key={item.step} to={item.href} className="workflow-step"><span className="workflow-number">{item.step}</span><div><strong>{item.title}</strong><p>{item.detail}</p></div><item.icon size={16} /></Link>)}
        <div className="workflow-note"><span /><span>Every experiment is a step toward better inference.</span></div>
      </section>
    </div>
  </div>
}
