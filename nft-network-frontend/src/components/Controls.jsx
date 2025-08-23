export default function Controls({ 
  layout, 
  setLayout, 
  minDegree, 
  setMinDegree, 
  colorBy, 
  setColorBy, 
  dataset, 
  setDataset, 
  datasets = [], 
  nodeSpacing, 
  setNodeSpacing,
  density,
  setDensity,
  normalized,
  setNormalized
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-3 shadow-sm">
      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Dataset</label>
        <select
          className="rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
        >
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>{d.label}</option>
          ))}
        </select>
      </div>
      
      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Layout</label>
        <select
          className="rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          value={layout}
          onChange={(e) => setLayout(e.target.value)}
        >
          <option value="graph">Graph</option>
          <option value="bubble">Bubble Map</option>
        </select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Density</label>
        <select
          className="rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          value={density}
          onChange={(e) => setDensity(e.target.value)}
        >
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Edge Type</label>
        <select
          className="rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          value={normalized ? 'normalized' : 'raw'}
          onChange={(e) => setNormalized(e.target.value === 'normalized')}
        >
          <option value="raw">Raw Holders</option>
          <option value="normalized">Jaccard Similarity</option>
        </select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Min Degree</label>
        <input
          type="range"
          min={0}
          max={50}
          value={minDegree}
          onChange={(e) => setMinDegree(Number(e.target.value))}
          className="text-gray-900 dark:text-white"
        />
        <span className="text-xs text-gray-500 dark:text-gray-400 w-8 text-right">{minDegree}</span>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600 dark:text-gray-400">Node Spacing</label>
        <input
          type="range"
          min={60}
          max={300}
          step={20}
          value={nodeSpacing}
          onChange={(e) => setNodeSpacing(Number(e.target.value))}
          className="text-gray-900 dark:text-white"
        />
        <span className="text-xs text-gray-500 dark:text-gray-400 w-12 text-right">{nodeSpacing}px</span>
      </div>
    </div>
  );
}


