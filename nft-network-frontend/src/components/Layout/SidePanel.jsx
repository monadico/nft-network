import { useEffect, useRef, useState } from 'react';

export default function SidePanel({ node, degrees = 0, onClose }) {
  const [pos, setPos] = useState({ x: 40, y: 80 });
  const [dragging, setDragging] = useState(false);
  const offsetRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging) return;
      setPos((p) => ({ x: e.clientX - offsetRef.current.x, y: e.clientY - offsetRef.current.y }));
    };
    const onUp = () => setDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging]);

  const startDrag = (e) => {
    const rect = e.currentTarget.parentElement.getBoundingClientRect();
    offsetRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setDragging(true);
  };

  if (!node) return null;

  return (
    <div
      className="fixed z-30 w-96 cursor-default rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-xl"
      style={{ left: pos.x, top: pos.y }}
    >
      <div
        className="flex cursor-move select-none items-center justify-between rounded-t-lg border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 p-3"
        onMouseDown={startDrag}
      >
        <h2 className="text-sm font-semibold text-gray-900 dark:text-white">Node Details</h2>
        <button onClick={onClose} className="rounded border border-gray-300 dark:border-gray-600 px-2 py-1 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600">Close</button>
      </div>
      <div className="space-y-3 p-4 text-sm select-text text-gray-900 dark:text-white">
        <div>
          <div className="text-gray-500 dark:text-gray-400">Name</div>
          <div className="break-all">{node.name || node.id}</div>
        </div>
        
        <div className="grid grid-cols-2 gap-2">
          <div>
            <div className="text-gray-500 dark:text-gray-400">Community</div>
            <div className="font-medium">{node.community ?? 'n/a'}</div>
          </div>
          <div>
            <div className="text-gray-500 dark:text-gray-400">Degree</div>
            <div>{degrees}</div>
          </div>
        </div>

        <div>
          <div className="text-gray-500 dark:text-gray-400 mb-2">Centrality Metrics</div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span>Degree Centrality:</span>
              <span className="font-mono">{(node.degree_centrality || 0).toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Eigenvector Centrality:</span>
              <span className="font-mono">{(node.eigenvector_centrality || 0).toFixed(4)}</span>
            </div>
          </div>
        </div>

        <div>
          <div className="text-gray-500 dark:text-gray-400">Holder Count</div>
          <div className="font-medium">{node.size || 'n/a'}</div>
        </div>
      </div>
    </div>
  );
}


