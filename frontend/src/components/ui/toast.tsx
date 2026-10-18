import * as React from 'react'
import { cn } from '../lib/utils'

interface ToastProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'destructive' | 'success'
}

const Toast = React.forwardRef<HTMLDivElement, ToastProps>(
  ({ className, variant = 'default', children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'rounded-lg border p-4 shadow-lg animate-in slide-in-from-top-full sm:slide-in-from-bottom-full',
        {
          'border-destructive bg-destructive/10 text-destructive': variant === 'destructive',
          'border-green-500 bg-green-500/10 text-green-500': variant === 'success',
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

interface ToastProviderProps {
  children: React.ReactNode
}

const toasts: Array<{ id: string; message: React.ReactNode; variant: ToastProps['variant'] }> = []
const listeners: Array<(toasts: typeof toasts) => void> = []

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
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-80">
      {toasts.map((t) => (
        <Toast key={t.id} variant={t.variant}>
          {t.message}
        </Toast>
      ))}
    </div>
  )
}

export { Toast }