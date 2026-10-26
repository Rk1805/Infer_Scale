import * as React from 'react'
import { cn } from '../../lib/utils'

interface TabsProps extends React.HTMLAttributes<HTMLDivElement> {
  defaultValue: string
  onValueChange?: (value: string) => void
}

const Tabs = React.forwardRef<HTMLDivElement, TabsProps>(
  ({ className, defaultValue, onValueChange, children, ...props }, ref) => {
    const [value, setValue] = React.useState(defaultValue)
    
    const handleChange = (newValue: string) => {
      setValue(newValue)
      onValueChange?.(newValue)
    }

    return (
      <div ref={ref} className={cn('space-y-4', className)} {...props}>
        <TabsContext.Provider value={{ value, onValueChange: handleChange }}>
          {children}
        </TabsContext.Provider>
      </div>
    )
  }
)
Tabs.displayName = 'Tabs'

const TabsContext = React.createContext<{ value: string; onValueChange: (value: string) => void } | null>(null)

function useTabsContext() {
  const context = React.useContext(TabsContext)
  if (!context) {
    throw new Error('Tabs components must be used within Tabs')
  }
  return context
}

interface TabsListProps extends React.HTMLAttributes<HTMLDivElement> {}

const TabsList = React.forwardRef<HTMLDivElement, TabsListProps>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('inline-flex min-h-10 max-w-full items-center gap-1 overflow-x-auto rounded-lg border bg-muted/60 p-1 text-muted-foreground', className)}
      {...props}
    >
      {children}
    </div>
  )
)
TabsList.displayName = 'TabsList'

interface TabsTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string
}

const TabsTrigger = React.forwardRef<HTMLButtonElement, TabsTriggerProps>(
  ({ className, value, children, ...props }, ref) => {
    const { value: currentValue, onValueChange } = useTabsContext()
    const isActive = currentValue === value
    
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50',
          isActive ? 'bg-card text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground',
          className
        )}
        onClick={() => onValueChange(value)}
        type="button"
        aria-pressed={isActive}
        {...props}
      >
        {children}
      </button>
    )
  }
)
TabsTrigger.displayName = 'TabsTrigger'

interface TabsContentProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string
}

const TabsContent = React.forwardRef<HTMLDivElement, TabsContentProps>(
  ({ className, value, children, ...props }, ref) => {
    const { value: currentValue } = useTabsContext()
    
    if (currentValue !== value) return null
    
    return (
      <div
        ref={ref}
        className={cn('mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2', className)}
        {...props}
      >
        {children}
      </div>
    )
  }
)
TabsContent.displayName = 'TabsContent'

export { Tabs, TabsList, TabsTrigger, TabsContent }
