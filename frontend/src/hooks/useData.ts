import { useState, useEffect, useCallback } from 'react'
import { api, ExperimentJob, ExperimentMetrics, TimelineEvent, TrainingJob, ModelInfo, Dataset, DatasetStats } from './api'

export function useHealth() {
  const [health, setHealth] = useState<{ status: string; timestamp: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchHealth = useCallback(async () => {
    try {
      setLoading(true)
      const data = await api.health()
      setHealth(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch health')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30000)
    return () => clearInterval(interval)
  }, [fetchHealth])

  return { health, loading, error, refetch: fetchHealth }
}

export function useExperiments(status?: string) {
  const [experiments, setExperiments] = useState<ExperimentJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchExperiments = useCallback(async () => {
    try {
      setLoading(true)
      const data = await api.listExperiments(status)
      setExperiments(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch experiments')
    } finally {
      setLoading(false)
    }
  }, [status])

  useEffect(() => {
    fetchExperiments()
  }, [fetchExperiments])

  return { experiments, loading, error, refetch: fetchExperiments }
}

export function useExperiment(id: string | null) {
  const [experiment, setExperiment] = useState<ExperimentJob | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchExperiment = useCallback(async () => {
    if (!id) return
    try {
      setLoading(true)
      const data = await api.getExperiment(id)
      setExperiment(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch experiment')
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    fetchExperiment()
    if (id) {
      const interval = setInterval(fetchExperiment, 2000)
      return () => clearInterval(interval)
    }
  }, [fetchExperiment, id])

  return { experiment, loading, error, refetch: fetchExperiment }
}

export function useExperimentMetrics(id: string | null) {
  const [metrics, setMetrics] = useState<ExperimentMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = useCallback(async () => {
    if (!id) return
    try {
      setLoading(true)
      const data = await api.getExperimentMetrics(id)
      setMetrics(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics')
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    fetchMetrics()
  }, [fetchMetrics])

  return { metrics, loading, error, refetch: fetchMetrics }
}

export function useExperimentTimeline(id: string | null) {
  const [timeline, setTimeline] = useState<TimelineEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTimeline = useCallback(async () => {
    if (!id) return
    try {
      setLoading(true)
      const data = await api.getExperimentTimeline(id)
      setTimeline(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch timeline')
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    fetchTimeline()
  }, [fetchTimeline])

  return { timeline, loading, error, refetch: fetchTimeline }
}

export function useTraining(id: string | null) {
  const [training, setTraining] = useState<TrainingJob | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTraining = useCallback(async () => {
    if (!id) return
    try {
      setLoading(true)
      const data = await api.getTraining(id)
      setTraining(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch training')
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    fetchTraining()
    if (id) {
      const interval = setInterval(fetchTraining, 1000)
      return () => clearInterval(interval)
    }
  }, [fetchTraining, id])

  return { training, loading, error, refetch: fetchTraining }
}

export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true)
      const data = await api.listModels()
      setModels(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  return { models, loading, error, refetch: fetchModels }
}

export function useDatasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchDatasets = useCallback(async () => {
    try {
      setLoading(true)
      const data = await api.listDatasets()
      setDatasets(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch datasets')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchDatasets()
  }, [fetchDatasets])

  return { datasets, loading, error, refetch: fetchDatasets }
}

export function useDatasetStats(id: string | null) {
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    if (!id) return
    try {
      setLoading(true)
      const data = await api.getDatasetStats(id)
      setStats(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch dataset stats')
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  return { stats, loading, error, refetch: fetchStats }
}