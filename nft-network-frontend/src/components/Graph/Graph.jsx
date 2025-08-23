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

export default function Graph({ data, width = undefined, height = undefined, minDegree = 0, nodeSpacing = 120, onSelectNode }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const [dimensions, setDimensions] = useState({ w: 0, h: 0 });
  const { isDarkMode } = useDarkMode();

  const { nodes, links } = useMemo(() => ({
    nodes: data?.nodes?.map((n) => ({ ...n })) ?? [],
    links: data?.edges?.map((e) => ({ ...e })) ?? [],
  }), [data]);

  const { filteredNodes, filteredLinks, degrees, colorFor } = useMemo(() => {
    const deg = computeNodeDegree(nodes, links);
    const keepNodes = nodes.filter((n) => (deg.get(n.id) || 0) >= minDegree);
    const idSet = new Set(keepNodes.map((n) => n.id));
    const keepLinks = links.filter((l) => {
      const src = typeof l.source === 'object' ? l.source?.id : l.source;
      const tgt = typeof l.target === 'object' ? l.target?.id : l.target;
      return idSet.has(src) && idSet.has(tgt);
    });

    // Use Louvain communities for colors (primary clustering method)
    const palette = d3.scaleOrdinal(d3.schemeTableau10);
    const colorForNode = (n) => {
      // Always use Louvain community for colors
      return palette(n.community ?? 0);
    };

    return { filteredNodes: keepNodes, filteredLinks: keepLinks, degrees: deg, colorFor: colorForNode };
  }, [nodes, links, minDegree]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    const resize = () => {
      const rect = container.getBoundingClientRect();
      setDimensions({ w: width ?? rect.width, h: height ?? rect.height });
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [width, height]);

  useEffect(() => {
    if (!svgRef.current || filteredNodes.length === 0 || filteredLinks.length === 0) return;

    const w = dimensions.w || 800;
    const h = dimensions.h || 600;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    svg.attr('viewBox', [0, 0, w, h]).attr('width', w).attr('height', h);

    const g = svg.append('g');

    // Dynamic colors based on dark mode
    const linkColor = isDarkMode ? '#6b7280' : '#d1d5db'; // gray-500 for dark, gray-300 for light
    const nodeStrokeColor = isDarkMode ? '#374151' : '#ffffff'; // gray-700 for dark, white for light
    const tooltipBgColor = isDarkMode ? '#1f2937' : '#ffffff'; // gray-800 for dark, white for light
    const tooltipBorderColor = isDarkMode ? '#4b5563' : '#e5e7eb'; // gray-600 for dark, gray-200 for light
    const tooltipTextColor = isDarkMode ? '#d1d5db' : '#6b7280'; // gray-300 for dark, gray-500 for light

    // Tooltip
    const tooltip = d3
      .select(containerRef.current)
      .append('div')
      .attr('class', 'absolute z-10 rounded px-2 py-1 text-xs shadow border hidden')
      .style('background-color', tooltipBgColor)
      .style('border-color', tooltipBorderColor)
      .style('color', isDarkMode ? '#f9fafb' : '#111827') // gray-50 for dark, gray-900 for light
      .style('pointer-events', 'none');

    const degrees = computeNodeDegree(filteredNodes, filteredLinks);

    // Prepare simulation
    const linkForce = d3
      .forceLink(filteredLinks)
      .id((d) => d.id)
      .distance((d) => {
        const distance = d.weight ? nodeSpacing / Math.log2(2 + d.weight) : nodeSpacing;
        return distance;
      })
      .strength(0.3);

    const simulation = d3
      .forceSimulation(filteredNodes)
      .force('link', linkForce)
      .force('charge', d3.forceManyBody().strength(-nodeSpacing * 0.7))
      .force('center', d3.forceCenter(w / 2, h / 2))
      .force('collision', d3.forceCollide().radius((d) => Math.max(8, nodeSpacing * 0.1 + Math.sqrt(degrees.get(d.id) || 1))));
    
    // Restart simulation to apply new forces
    simulation.alpha(1).restart();

    // Draw links
    const link = g
      .append('g')
      .attr('stroke', linkColor)
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(filteredLinks)
      .join('line')
      .attr('stroke-width', (d) => Math.max(0.5, Math.min(2, d.weight ? Math.log2(1.5 + d.weight) * 0.5 : 0.5)));

    // Draw nodes - size based on degree centrality
    const node = g
      .append('g')
      .attr('stroke', nodeStrokeColor)
      .attr('stroke-width', 1)
      .selectAll('circle')
      .data(filteredNodes)
      .join('circle')
      .attr('r', (d) => {
        // Use degree centrality for node size, with fallback to holder count
        const centrality = d.degree_centrality || 0;
        const baseSize = 3 + Math.min(6, Math.sqrt(degrees.get(d.id) || 1));
        const centralityMultiplier = 1 + centrality * 2; // Scale by centrality
        return baseSize * centralityMultiplier;
      })
      .attr('fill', (d) => colorFor(d))
      .call(
        d3
          .drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      )
      .on('mouseover', function (event, d) {
        const degree = degrees.get(d.id) || 0;
        tooltip
          .classed('hidden', false)
          .html(
            `<div class="font-medium">${d.name || d.id}</div>` +
              `<div style="color: ${tooltipTextColor}">Community: ${d.community ?? 'n/a'}</div>` +
              `<div style="color: ${tooltipTextColor}">Degree: ${degree}</div>` +
              `<div style="color: ${tooltipTextColor}">Degree Centrality: ${(d.degree_centrality || 0).toFixed(3)}</div>` +
              `<div style="color: ${tooltipTextColor}">Eigenvector Centrality: ${(d.eigenvector_centrality || 0).toFixed(3)}</div>` +
              `<div style="color: ${tooltipTextColor}">Holders: ${d.size || 'n/a'}</div>`
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
        if (onSelectNode) onSelectNode({ ...d }, degrees.get(d.id) || 0);
      });

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);
    });

    // Zoom and pan
    const zoomed = (event) => {
      g.attr('transform', event.transform);
    };
    const zoom = d3.zoom().scaleExtent([0.2, 6]).on('zoom', zoomed);
    svg.call(zoom);

    return () => {
      simulation.stop();
      tooltip.remove();
    };
  }, [nodes, links, dimensions, minDegree, nodeSpacing, isDarkMode]);

  return (
    <div ref={containerRef} className="relative h-full w-full">
      <svg ref={svgRef} className="h-full w-full select-none" />
    </div>
  );
}


