import { useState } from 'react';
import { useDataLoader } from '../hooks/useDataLoader';
import BubbleMap from '../components/Graph/BubbleMap';
import Controls from '../components/Controls';

export default function BubbleViewer() {
  const [density, setDensity] = useState('medium');
  const [normalized, setNormalized] = useState(false);
  const [minDegree, setMinDegree] = useState(0);
  const [nodeSpacing, setNodeSpacing] = useState(120);
  
  // Get dataset key based on density and normalized settings
  const datasetKey = normalized ? `${density}_density_normalized` : `${density}_density`;
  const { data, loading, error } = useDataLoader(datasetKey);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <h1 className="text-xl font-semibold text-gray-900 dark:text-white">Bubble Map</h1>
      <div className="mt-3">
        <Controls 
          layout="bubble" 
          setLayout={() => {}} 
          minDegree={minDegree} 
          setMinDegree={setMinDegree} 
          dataset={datasetKey}
          setDataset={() => {}}
          datasets={[]}
          nodeSpacing={nodeSpacing}
          setNodeSpacing={setNodeSpacing}
          density={density}
          setDensity={setDensity}
          normalized={normalized}
          setNormalized={setNormalized}
        />
      </div>
      {loading && <p className="mt-4 text-gray-600 dark:text-gray-400">Loading dataset…</p>}
      {error && <p className="mt-4 text-red-600 dark:text-red-400">Failed to load: {String(error.message || error)}</p>}
      {data && (
        <div className="mt-4">
          <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>Nodes: {data.nodes?.length ?? 0}</span>
            <span>Edges: {data.edges?.length ?? 0}</span>
            <span>Threshold: {data.metadata?.threshold ?? 'N/A'}</span>
            <span>Type: {data.metadata?.normalized ? 'Jaccard Similarity' : 'Raw Holders'}</span>
          </div>
          <div className="mt-2 h-[70vh] w-full overflow-hidden rounded border border-gray-200 dark:border-gray-700">
            <BubbleMap data={data} minDegree={minDegree} />
          </div>
        </div>
      )}
    </div>
  );
}


