export const DATASETS = [
  { id: 'low_density', label: 'Low Density', normalized: false, path: '/src/data/low_density.json' },
  { id: 'medium_density', label: 'Medium Density', normalized: false, path: '/src/data/medium_density.json' },
  { id: 'high_density', label: 'High Density', normalized: false, path: '/src/data/high_density.json' },
  { id: 'low_density_normalized', label: 'Low Density (Normalized)', normalized: true, path: '/src/data/low_density_normalized.json' },
  { id: 'medium_density_normalized', label: 'Medium Density (Normalized)', normalized: true, path: '/src/data/medium_density_normalized.json' },
  { id: 'high_density_normalized', label: 'High Density (Normalized)', normalized: true, path: '/src/data/high_density_normalized.json' },
  { id: 'nft_network', label: 'NFT Network (Full)', normalized: false, path: '/src/data/nft_network.json' },
];

export async function fetchLocalGraph(dataset = 'medium_density') {
  const entry = DATASETS.find((d) => d.id === dataset);
  const path = entry?.path;

  if (!path) {
    throw new Error(`Unknown dataset: ${dataset}`);
  }

  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load ${dataset}: ${response.status}`);
  }
  const json = await response.json();
  return json;
}


