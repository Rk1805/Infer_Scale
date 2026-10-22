import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { cn } from '../lib/utils'
import { LayoutDashboard, FlaskConical, Activity, GitCompare, Brain, Database, History } from 'lucide-react'
import { ToastContainer } from './ui/toast'

const navigation = [
  { name: 'Overview', href: '/', icon: LayoutDashboard },
  { name: 'Experiment Studio', href: '/experiment-studio', icon: FlaskConical },
  { name: 'Simulation', href: '/simulation', icon: Activity },
  { name: 'Policy Comparison', href: '/policy-comparison', icon: GitCompare },
  { name: 'Training Lab', href: '/training-lab', icon: Brain },
  { name: 'Dataset Explorer', href: '/dataset-explorer', icon: Database },
  { name: 'History & Reports', href: '/experiment-history', icon: History },
]

export function Layout() {
  const location = useLocation()
  
  return (
    <div className="min-h-screen bg-background flex">
      <aside className="w-64 border-r bg-card flex flex-col">
        <div className="p-4 border-b">
          <h1 className="text-xl font-bold text-primary">InferScale</h1>
          <p className="text-xs text-muted-foreground">LLM Inference Autoscaling</p>
        </div>
        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.href || (item.href !== '/' && location.pathname.startsWith(item.href))
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
              >
                <Icon className="h-4 w-4" />
                {item.name}
              </NavLink>
            )
          })}
        </nav>
        <div className="p-4 border-t">
          <div className="flex items-center gap-3 px-3 py-2 text-xs text-muted-foreground">
            <span className="flex h-2 w-2 rounded-full bg-green-500" />
            <span>API Connected</span>
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
      <ToastContainer />
    </div>
  )
}