const labelData = [
  { name: 'Normal', value: 9045, color: '#0f766e' },
  { name: 'Abnormal', value: 5818, color: '#ad7d3d' },
];

const modelData = [
  { name: 'CNN', value: 4, color: '#183b56' },
  { name: 'VGG', value: 2, color: '#5b6fd4' },
  { name: 'ResNet', value: 2, color: '#c06e4a' },
  { name: 'DenseNet', value: 5, color: '#0f766e' },
];

function createSvg(width, height) {
  return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" role="img" aria-label="chart">
      <defs>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#13222d" flood-opacity="0.14" />
        </filter>
      </defs>
    </svg>
  `;
}

function renderDonutChart(target, data) {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const width = 600;
  const height = 360;
  const centerX = 185;
  const centerY = 180;
  const radius = 100;
  const strokeWidth = 34;
  const circumference = 2 * Math.PI * radius;
  let progress = 0;

  const container = document.querySelector(target);
  container.innerHTML = createSvg(width, height);
  const svg = container.querySelector('svg');

  const infoGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  infoGroup.setAttribute('transform', `translate(${centerX}, ${centerY})`);
  svg.appendChild(infoGroup);

  const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  title.setAttribute('text-anchor', 'middle');
  title.setAttribute('y', '-8');
  title.setAttribute('font-size', '18');
  title.setAttribute('font-weight', '700');
  title.setAttribute('fill', '#13222d');
  title.textContent = '14,863';
  infoGroup.appendChild(title);

  const subtitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  subtitle.setAttribute('text-anchor', 'middle');
  subtitle.setAttribute('y', '22');
  subtitle.setAttribute('font-size', '12');
  subtitle.setAttribute('fill', '#5c6f78');
  subtitle.textContent = 'studies total';
  infoGroup.appendChild(subtitle);

  const ringBackground = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  ringBackground.setAttribute('cx', centerX);
  ringBackground.setAttribute('cy', centerY);
  ringBackground.setAttribute('r', radius);
  ringBackground.setAttribute('fill', 'none');
  ringBackground.setAttribute('stroke', 'rgba(19,34,45,0.08)');
  ringBackground.setAttribute('stroke-width', strokeWidth);
  svg.appendChild(ringBackground);

  data.forEach((item, index) => {
    const segment = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    segment.setAttribute('cx', centerX);
    segment.setAttribute('cy', centerY);
    segment.setAttribute('r', radius);
    segment.setAttribute('fill', 'none');
    segment.setAttribute('stroke', item.color);
    segment.setAttribute('stroke-width', strokeWidth);
    segment.setAttribute('stroke-linecap', 'round');
    segment.setAttribute('stroke-dasharray', `${circumference * (item.value / total)} ${circumference}`);
    segment.setAttribute('stroke-dashoffset', `${-progress}`);
    segment.setAttribute('transform', `rotate(-90 ${centerX} ${centerY})`);
    svg.appendChild(segment);
    progress += circumference * (item.value / total);
  });

  const axisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  axisLabel.setAttribute('x', 365);
  axisLabel.setAttribute('y', 72);
  axisLabel.setAttribute('font-size', '13');
  axisLabel.setAttribute('font-weight', '700');
  axisLabel.setAttribute('fill', '#13222d');
  axisLabel.textContent = 'Dataset balance';
  svg.appendChild(axisLabel);

  data.forEach((item, index) => {
    const rowY = 118 + index * 72;
    const swatch = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    swatch.setAttribute('x', 365);
    swatch.setAttribute('y', rowY - 16);
    swatch.setAttribute('width', 14);
    swatch.setAttribute('height', 14);
    swatch.setAttribute('rx', 7);
    swatch.setAttribute('fill', item.color);
    svg.appendChild(swatch);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', 388);
    label.setAttribute('y', rowY - 5);
    label.setAttribute('font-size', '15');
    label.setAttribute('fill', '#13222d');
    label.textContent = item.name;
    svg.appendChild(label);

    const value = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    value.setAttribute('x', 540);
    value.setAttribute('y', rowY - 5);
    value.setAttribute('font-size', '15');
    value.setAttribute('font-weight', '700');
    value.setAttribute('text-anchor', 'end');
    value.setAttribute('fill', '#13222d');
    value.textContent = item.value.toLocaleString();
    svg.appendChild(value);

    const barBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    barBg.setAttribute('x', 388);
    barBg.setAttribute('y', rowY + 5);
    barBg.setAttribute('width', 150);
    barBg.setAttribute('height', 10);
    barBg.setAttribute('rx', 5);
    barBg.setAttribute('fill', 'rgba(19,34,45,0.08)');
    svg.appendChild(barBg);

    const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bar.setAttribute('x', 388);
    bar.setAttribute('y', rowY + 5);
    bar.setAttribute('width', 150 * (item.value / Math.max(...data.map((d) => d.value))));
    bar.setAttribute('height', 10);
    bar.setAttribute('rx', 5);
    bar.setAttribute('fill', item.color);
    svg.appendChild(bar);
  });
}

function renderBarChart(target, data) {
  const width = 600;
  const height = 360;
  const container = document.querySelector(target);
  container.innerHTML = createSvg(width, height);
  const svg = container.querySelector('svg');

  const chartLeft = 64;
  const chartTop = 44;
  const chartWidth = 480;
  const chartHeight = 250;
  const maxValue = Math.max(...data.map((item) => item.value));
  const barWidth = 72;
  const gap = 34;

  for (let tick = 0; tick <= 5; tick += 1) {
    const y = chartTop + chartHeight - (chartHeight / 5) * tick;
    const grid = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    grid.setAttribute('x1', chartLeft);
    grid.setAttribute('x2', chartLeft + chartWidth);
    grid.setAttribute('y1', y);
    grid.setAttribute('y2', y);
    grid.setAttribute('stroke', 'rgba(19,34,45,0.08)');
    svg.appendChild(grid);
  }

  data.forEach((item, index) => {
    const x = chartLeft + index * (barWidth + gap);
    const heightPx = (item.value / maxValue) * chartHeight;
    const y = chartTop + chartHeight - heightPx;

    const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bar.setAttribute('x', x);
    bar.setAttribute('y', y);
    bar.setAttribute('width', barWidth);
    bar.setAttribute('height', heightPx);
    bar.setAttribute('rx', 18);
    bar.setAttribute('fill', item.color);
    bar.setAttribute('filter', 'url(#shadow)');
    svg.appendChild(bar);

    const value = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    value.setAttribute('x', x + barWidth / 2);
    value.setAttribute('y', y - 10);
    value.setAttribute('text-anchor', 'middle');
    value.setAttribute('font-size', '14');
    value.setAttribute('font-weight', '700');
    value.setAttribute('fill', '#13222d');
    value.textContent = item.value;
    svg.appendChild(value);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', x + barWidth / 2);
    label.setAttribute('y', chartTop + chartHeight + 28);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-size', '14');
    label.setAttribute('fill', '#13222d');
    label.textContent = item.name;
    svg.appendChild(label);
  });

  const axisTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  axisTitle.setAttribute('x', chartLeft);
  axisTitle.setAttribute('y', 24);
  axisTitle.setAttribute('font-size', '13');
  axisTitle.setAttribute('font-weight', '700');
  axisTitle.setAttribute('fill', '#13222d');
  axisTitle.textContent = 'Model families represented in the notebook';
  svg.appendChild(axisTitle);

  const note = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  note.setAttribute('x', chartLeft + chartWidth);
  note.setAttribute('y', 24);
  note.setAttribute('font-size', '11');
  note.setAttribute('text-anchor', 'end');
  note.setAttribute('fill', '#5c6f78');
  note.textContent = '13 model sections total';
  svg.appendChild(note);
}

renderDonutChart('#labelChart', labelData);
renderBarChart('#modelChart', modelData);
