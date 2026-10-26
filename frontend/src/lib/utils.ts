import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(value: number | null | undefined, digits = 1) {
  return value == null ? 'N/A' : `${(value * 100).toFixed(digits)}%`
}
