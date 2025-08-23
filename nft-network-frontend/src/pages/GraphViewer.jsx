import { useDataLoader } from '../hooks/useDataLoader';
import Graph from '../components/Graph/Graph';
import { useState } from 'react';
import Controls from '../components/Controls';
import BubbleMap from '../components/Graph/BubbleMap';
import { DATASETS } from '../services/dataService';
import SidePanel from '../components/Layout/SidePanel';

export default function GraphViewer() {
  const [dataset, setDataset] = useState('medium_density');
  const { data, loading, error } = useDataLoader(dataset);
  const [layout, setLayout] = useState('graph');
  const [minDegree, setMinDegree] = useState(0);
  const [density, setDensity] = useState('medium');
  const [normalized, setNormalized] = useState(false);
  const [nodeSpacing, setNodeSpacing] = useState(120);
  const [selected, setSelected] = useState({ node: null, degree: 0 });

  // Update dataset when density or normalized changes
  const getDatasetKey = () => {
    if (normalized) {
      return `${density}_density_normalized`;
    } else {
      return `${density}_density`;
    }
  };

  const currentDataset = getDatasetKey();

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <h1 className="text-xl font-semibold text-gray-900 dark:text-white">Graph Viewer</h1>
      <div className="mt-3">
        <Controls
          layout={layout}
          setLayout={setLayout}
          minDegree={minDegree}
          setMinDegree={setMinDegree}
          dataset={currentDataset}
          setDataset={setDataset}
          datasets={DATASETS}
          nodeSpacing={nodeSpacing}
          setNodeSpacing={setNodeSpacing}
          density={density}
          setDensity={setDensity}
          normalized={normalized}
          setNormalized={setNormalized}
        />
      </div>
      {loading && (
        <p className="mt-4 text-gray-600 dark:text-gray-400">Loading dataset…</p>
      )}
      {error && (
        <p className="mt-4 text-red-600 dark:text-red-400">Failed to load: {String(error.message || error)}</p>
      )}
      {data && (
        <div className="mt-4">
          <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>Nodes: {data.nodes?.length ?? 0}</span>
            <span>Edges: {data.edges?.length ?? 0}</span>
            <span>Threshold: {data.metadata?.threshold ?? 'N/A'}</span>
            <span>Type: {data.metadata?.normalized ? 'Jaccard Similarity' : 'Raw Holders'}</span>
          </div>
          <div className="relative mt-2 h-[70vh] w-full rounded border border-gray-200 dark:border-gray-700 overflow-hidden">
            {layout === 'graph' ? (
              <Graph 
                data={data} 
                minDegree={minDegree} 
                nodeSpacing={nodeSpacing} 
                onSelectNode={(node, degree) => setSelected({ node, degree })} 
              />
            ) : (
              <BubbleMap 
                data={data} 
                minDegree={minDegree} 
                onSelectNode={(node, degree) => setSelected({ node, degree })} 
              />
            )}
            <SidePanel node={selected.node} degrees={selected.degree} onClose={() => setSelected({ node: null, degree: 0 })} />
          </div>
        </div>
      )}
    </div>
  );
}


