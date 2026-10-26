import * as React from 'react'
import { cn } from '../../lib/utils'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline'
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', ...props }, ref) => (
    <span
      ref={ref}
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[14px] font-semibold leading-5',
        {
          'border-primary/10 bg-primary/10 text-primary': variant === 'default',
          'border-border bg-secondary text-secondary-foreground': variant === 'secondary',
          'border-destructive/15 bg-destructive/10 text-destructive': variant === 'destructive',
          'border-border bg-card text-muted-foreground': variant === 'outline',
        },
        className
      )}
      {...props}
    />
  )
)
Badge.displayName = 'Badge'

export { Badge }
