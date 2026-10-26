import * as React from 'react';
import { cn } from '../../lib/utils';

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  onValueChange?: (value: string) => void
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, children, onValueChange, onChange, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        'field-control select-control',
        className
      )}
      {...props}
      onChange={event => { onChange?.(event); onValueChange?.(event.target.value) }}
    >
      {children}
    </select>
  )
)
Select.displayName = 'Select'

export { Select }
