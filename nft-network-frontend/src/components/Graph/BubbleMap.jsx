import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useDarkMode } from '../../contexts/DarkModeContext';

function computeNodeDegree(nodes, links) {
  const idToDegree = new Map();
  for (const n of nodes) idToDegree.set(n.id, 0);
  for (const l of links) {
    const src = typeof l.source === 'object' ? l.source?.id : l.source;
    const tgt = typeof l.target === 'object' ? l.target?.id : l.target;
    if (idToDegree.has(src)) idToDegree.set(src, (idToDegree.get(src) || 0) + 1);
    if (idToDegree.has(tgt)) idToDegree.set(tgt, (idToDegree.get(tgt) || 0) + 1);
  }
  return idToDegree;
}

export default function BubbleMap({ data, width = undefined, height = undefined, minDegree = 0, onSelectNode }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const [dimensions, setDimensions] = useState({ w: 0, h: 0 });
  const { isDarkMode } = useDarkMode();

  const nodes = data?.nodes ?? [];
  const links = data?.edges ?? [];

  const degrees = useMemo(() => computeNodeDegree(nodes, links), [nodes, links]);

  // Build hierarchical data: groups by Louvain community
  const rootData = useMemo(() => {
    const groups = d3.group(nodes.filter((n) => (degrees.get(n.id) || 0) >= minDegree), (n) => n.community ?? 'ungrouped');
    const children = Array.from(groups, ([key, arr]) => ({
      name: `Community ${key}`,
      children: arr.map((n) => ({
        ...n,
        value: Math.max(1, degrees.get(n.id) || 1),
      })),
    }));
    return { name: 'root', children };
  }, [nodes, degrees, minDegree]);

  useEffect(() => {
    if (!containerRef.current) return;
    const resize = () => {
      const rect = containerRef.current.getBoundingClientRect();
      setDimensions({ w: width ?? rect.width, h: height ?? rect.height });
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [width, height]);

  useEffect(() => {
    if (!svgRef.current || !rootData?.children?.length) return;
    const w = dimensions.w || 800;
    const h = dimensions.h || 600;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', [0, 0, w, h]).attr('width', w).attr('height', h);

    const g = svg.append('g');

    // Dynamic colors based on dark mode
    const tooltipBgColor = isDarkMode ? '#1f2937' : '#ffffff'; // gray-800 for dark, white for light
    const tooltipBorderColor = isDarkMode ? '#4b5563' : '#e5e7eb'; // gray-600 for dark, gray-200 for light
    const tooltipTextColor = isDarkMode ? '#d1d5db' : '#6b7280'; // gray-300 for dark, gray-500 for light
    const groupBorderColor = isDarkMode ? '#6b7280' : '#e5e7eb'; // gray-500 for dark, gray-200 for light
    const nodeStrokeColor = isDarkMode ? '#374151' : '#ffffff'; // gray-700 for dark, white for light
    const labelColor = isDarkMode ? '#9ca3af' : '#6b7280'; // gray-400 for dark, gray-500 for light

    const tooltip = d3
      .select(containerRef.current)
      .append('div')
      .attr('class', 'absolute z-10 rounded px-2 py-1 text-xs shadow border hidden')
      .style('background-color', tooltipBgColor)
      .style('border-color', tooltipBorderColor)
      .style('color', isDarkMode ? '#f9fafb' : '#111827') // gray-50 for dark, gray-900 for light
      .style('pointer-events', 'none');

    // Use Louvain communities for colors
    const palette = d3.scaleOrdinal(d3.schemeTableau10);

    const hierarchy = d3
      .hierarchy(rootData)
      .sum((d) => d.value || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const pack = d3.pack().size([w, h]).padding(4);
    const root = pack(hierarchy);

    const nodesSelection = g
      .selectAll('g')
      .data(root.descendants())
      .join('g')
      .attr('transform', (d) => `translate(${d.x},${d.y})`);

    // Draw group bubbles (depth 1) - Louvain communities
    nodesSelection
      .filter((d) => d.depth === 1)
      .append('circle')
      .attr('r', (d) => d.r)
      .attr('fill', (d) => d3.color(palette(d.data.name.split(' ')[1])).copy({ opacity: 0.12 }))
      .attr('stroke', groupBorderColor);

    // Draw leaf bubbles (depth 2) - individual NFTs
    const leaves = nodesSelection.filter((d) => d.depth === 2);
    leaves
      .append('circle')
      .attr('r', (d) => {
        // Use degree centrality for bubble size
        const centrality = d.data.degree_centrality || 0;
        const baseSize = d.r;
        const centralityMultiplier = 1 + centrality * 1.5; // Scale by centrality
        return baseSize * centralityMultiplier;
      })
      .attr('fill', (d) => palette(d.data.community ?? 0))
      .attr('stroke', nodeStrokeColor)
      .attr('stroke-width', 1)
      .on('mouseover', function (event, d) {
        const node = d.data;
        const degree = degrees.get(node.id) || 0;
        tooltip
          .classed('hidden', false)
          .html(
            `<div class="font-medium">${node.name || node.id}</div>` +
              `<div style="color: ${tooltipTextColor}">Community: ${node.community ?? 'n/a'}</div>` +
              `<div style="color: ${tooltipTextColor}">Degree: ${degree}</div>` +
              `<div style="color: ${tooltipTextColor}">Degree Centrality: ${(node.degree_centrality || 0).toFixed(3)}</div>` +
              `<div style="color: ${tooltipTextColor}">Eigenvector Centrality: ${(node.eigenvector_centrality || 0).toFixed(3)}</div>` +
              `<div style="color: ${tooltipTextColor}">Holders: ${node.size || 'n/a'}</div>`
          );
        d3.select(this).attr('stroke-width', 2);
      })
      .on('mousemove', function (event) {
        tooltip.style('left', event.offsetX + 12 + 'px').style('top', event.offsetY + 12 + 'px');
      })
      .on('mouseout', function () {
        tooltip.classed('hidden', true);
        d3.select(this).attr('stroke-width', 1);
      })
      .on('click', function (event, d) {
        if (onSelectNode) onSelectNode({ ...d.data }, degrees.get(d.data.id) || 0);
      });

    // Labels for groups (Louvain communities)
    nodesSelection
      .filter((d) => d.depth === 1)
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('class', 'pointer-events-none text-xs')
      .style('fill', labelColor)
      .text((d) => d.data.name);

    const zoomed = (event) => {
      g.attr('transform', event.transform);
    };
    const zoom = d3.zoom().scaleExtent([0.4, 6]).on('zoom', zoomed);
    svg.call(zoom);

    return () => {
      tooltip.remove();
    };
  }, [rootData, degrees, dimensions, isDarkMode]);

  return (
    <div ref={containerRef} className="relative h-full w-full">
      <svg ref={svgRef} className="h-full w-full select-none" />
    </div>
  );
}


