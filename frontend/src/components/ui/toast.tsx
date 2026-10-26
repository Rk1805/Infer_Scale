import * as React from 'react'
import { cn } from '../../lib/utils'

interface ToastProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'destructive' | 'success'
}

const Toast = React.forwardRef<HTMLDivElement, ToastProps>(
  ({ className, variant = 'default', children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'rounded-xl border bg-card p-4 text-sm shadow-xl',
        {
          'border-destructive/30 text-destructive': variant === 'destructive',
          'border-emerald-500/30 text-emerald-600 dark:text-emerald-400': variant === 'success',
          'border-border bg-background': variant === 'default',
        },
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
)
Toast.displayName = 'Toast'

type ToastEntry = { id: string; message: React.ReactNode; variant: ToastProps['variant'] }
const toasts: ToastEntry[] = []
const listeners: Array<(items: ToastEntry[]) => void> = []

function notify() {
  listeners.forEach((listener) => listener([...toasts]))
}

export function useToast() {
  const [, setToasts] = React.useState(toasts)
  
  React.useEffect(() => {
    listeners.push(setToasts)
    return () => {
      const idx = listeners.indexOf(setToasts)
      if (idx > -1) listeners.splice(idx, 1)
    }
  }, [])
  
  const toast = React.useCallback((message: React.ReactNode, variant: ToastProps['variant'] = 'default') => {
    const id = Math.random().toString(36).slice(2)
    toasts.push({ id, message, variant })
    notify()
    setTimeout(() => {
      const idx = toasts.findIndex((t) => t.id === id)
      if (idx > -1) {
        toasts.splice(idx, 1)
        notify()
      }
    }, 5000)
  }, [])
  
  return { toast, toasts }
}

export function ToastContainer() {
  const { toasts } = useToast()
  
  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2" aria-live="polite">
      {toasts.map((t) => (
        <Toast key={t.id} variant={t.variant}>
          {t.message}
        </Toast>
      ))}
    </div>
  )
}

export { Toast }
