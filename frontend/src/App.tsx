import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Overview } from './pages/Overview'
import { ExperimentStudio } from './pages/ExperimentStudio'
import { Simulation } from './pages/Simulation'
import { PolicyComparison } from './pages/PolicyComparison'
import { TrainingLab } from './pages/TrainingLab'
import { DatasetExplorer } from './pages/DatasetExplorer'
import { ExperimentHistory } from './pages/ExperimentHistory'

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Overview />} />
        <Route path="/experiment-studio" element={<ExperimentStudio />} />
        <Route path="/simulation/:id?" element={<Simulation />} />
        <Route path="/policy-comparison" element={<PolicyComparison />} />
        <Route path="/training-lab" element={<TrainingLab />} />
        <Route path="/dataset-explorer" element={<DatasetExplorer />} />
        <Route path="/experiment-history" element={<ExperimentHistory />} />
      </Route>
    </Routes>
  )
}

export default App