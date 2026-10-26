import { useEffect, useState } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { Activity, ArrowUpRight, BookOpen, Brain, ChevronRight, Command, Database, FlaskConical, GitCompare, History, LayoutDashboard, Menu, Moon, Sun, X } from 'lucide-react'
import { useHealth } from '../hooks/useData'
import { cn } from '../lib/utils'
import { ToastContainer } from './ui/toast'

const navigation = [
  { name: 'Overview', href: '/', icon: LayoutDashboard, group: 'Workspace' },
  { name: 'Experiment Studio', href: '/experiment-studio', icon: FlaskConical, group: 'Workspace' },
  { name: 'Simulation', href: '/simulation', icon: Activity, group: 'Workspace' },
  { name: 'Policy Comparison', href: '/policy-comparison', icon: GitCompare, group: 'Workspace' },
  { name: 'Training Lab', href: '/training-lab', icon: Brain, group: 'Resources' },
  { name: 'Dataset Explorer', href: '/dataset-explorer', icon: Database, group: 'Resources' },
  { name: 'History & Reports', href: '/experiment-history', icon: History, group: 'Resources' },
]

export function Layout() {
  const location = useLocation()
  const { health, error, loading } = useHealth()
  const connected = health?.status === 'ok' && !error
  const [mobileOpen, setMobileOpen] = useState(false)
  const [dark, setDark] = useState(() => {
    try { return localStorage.getItem('inferscale-theme') === 'dark' } catch { return false }
  })
  const current = navigation.find(item => item.href === '/' ? location.pathname === '/' : location.pathname.startsWith(item.href))

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try { localStorage.setItem('inferscale-theme', dark ? 'dark' : 'light') } catch { /* Storage may be disabled. */ }
  }, [dark])

  useEffect(() => {
    if (!mobileOpen) return
    const previousOverflow = document.body.style.overflow
    const previousFocus = document.activeElement as HTMLElement | null
    const panel = document.getElementById('workspace-navigation')
    const focusable = () => Array.from(panel?.querySelectorAll<HTMLElement>('a[href], button:not([disabled])') ?? [])
    document.body.style.overflow = 'hidden'
    focusable()[0]?.focus()
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMobileOpen(false)
      if (event.key === 'Tab') {
        const elements = focusable()
        const first = elements[0]
        const last = elements[elements.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }
    }
    const desktop = window.matchMedia('(min-width: 1024px)')
    const resize = () => { if (desktop.matches) setMobileOpen(false) }
    desktop.addEventListener('change', resize)
    window.addEventListener('keydown', close)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', close)
      desktop.removeEventListener('change', resize)
      previousFocus?.focus()
    }
  }, [mobileOpen])

  return <div className="workspace-shell">
    <a href="#workspace-main" className="skip-link">Skip to content</a>
    {mobileOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <aside id="workspace-navigation" className={cn('workspace-sidebar', mobileOpen && 'is-open')}>
      <NavLink to="/" className="brand-lockup" onClick={() => setMobileOpen(false)} aria-label="InferScale overview">
        <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
        <span className="brand-name">InferScale<span className="brand-version">RESEARCH</span></span>
      </NavLink>
      <div className="workspace-switcher">
        <span className="workspace-avatar"><Command size={17} /></span>
        <div><strong>Research workspace</strong><span>Local environment</span></div>
        <span className="workspace-local-dot" title="Local workspace" />
      </div>
      <nav aria-label="Main navigation" className="sidebar-navigation">
        {['Workspace', 'Resources'].map(group => <div className="navigation-group" key={group}>
          <p className="navigation-label">{group}</p>
          {navigation.filter(item => item.group === group).map(item => {
            const active = current?.href === item.href
            const Icon = item.icon
            return <NavLink key={item.href} to={item.href} end={item.href === '/'} onClick={() => setMobileOpen(false)}
              className={cn('navigation-item', active && 'is-active')}>
              <Icon size={18} strokeWidth={1.7} /><span>{item.name}</span>
              {active && <span className="navigation-active-dot" />}
            </NavLink>
          })}
        </div>)}
      </nav>
      <div className="sidebar-footer">
        <div className="sidebar-note"><span className="sidebar-note-icon"><Activity size={18} /></span>
          <strong>Built for better inference.</strong><p>Explore the balance between cost, capacity, and performance.</p></div>
        <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="sidebar-docs"><BookOpen size={16} /> API documentation <ArrowUpRight size={14} /></a>
        <div className="sidebar-profile"><span className="profile-avatar">IS</span><div><strong>InferScale Lab</strong><span>Research environment</span></div><span className="profile-version">v0.1</span></div>
      </div>
    </aside>
    <div className="workspace-body">
      <header className="workspace-topbar">
        <div className="flex min-w-0 items-center gap-3">
          <button className="icon-control mobile-menu-button" onClick={() => setMobileOpen(!mobileOpen)} aria-label={mobileOpen ? 'Close navigation' : 'Open navigation'} aria-expanded={mobileOpen} aria-controls="workspace-navigation">{mobileOpen ? <X size={20} /> : <Menu size={20} />}</button>
          <span className="topbar-workspace">Workspace</span><ChevronRight size={14} className="topbar-chevron" />
          <span className="topbar-current">{current?.name ?? 'Overview'}</span>
        </div>
        <div className="flex items-center gap-3 sm:gap-5">
          <span className={cn('connection-status', connected ? 'is-connected' : error && 'is-disconnected')} role="status">
            <span />{loading && !health ? 'Connecting' : connected ? 'API connected' : 'API unavailable'}
          </span>
          <span className="topbar-divider" />
          <button className="icon-control" onClick={() => setDark(!dark)} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'} title={dark ? 'Light mode' : 'Dark mode'}>{dark ? <Sun size={18} /> : <Moon size={18} />}</button>
        </div>
      </header>
      <main id="workspace-main" className="workspace-content" tabIndex={-1}><Outlet /></main>
      <footer className="workspace-footer"><span>InferScale <span className="footer-dot">/</span> Inference research platform</span><span>CPU simulation <span className="footer-dot">·</span> Reproducible by design</span></footer>
    </div>
    <ToastContainer />
  </div>
}
