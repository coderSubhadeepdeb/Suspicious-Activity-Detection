const API_BASE = 'http://localhost:8000/api';
let currentJobId = null;
let pollingInterval = null;
let uploadedFileURL = null;

// ── DOM refs ──
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const progressWrapper = document.getElementById('progressWrapper');
const progressLabel  = document.getElementById('progressLabel');
const progressBarFill = document.getElementById('progressBarFill');
const progressStatus = document.getElementById('progressStatus');
const uploadSection  = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');

// ── Drag & Drop ──
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File Handler ──
function handleFile(file) {
  const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
  if (!allowed.some(t => file.type.includes(t.split('/')[1])) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
    showToast('Invalid file type. Please upload MP4, AVI, MOV or MKV.', 'error');
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    showToast('File too large. Maximum size is 200 MB.', 'error');
    return;
  }
  uploadedFileURL = URL.createObjectURL(file);
  uploadFile(file);
}

// ── Upload ──
function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  progressWrapper.style.display = 'block';
  setProgress('Uploading video...', 0);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${API_BASE}/upload`);

  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      setProgress('Uploading...', pct);
    }
  });

  xhr.addEventListener('load', () => {
    if (xhr.status === 200) {
      const data = JSON.parse(xhr.responseText);
      currentJobId = data.job_id;
      setProgress('Upload complete. Processing...', 100);
      showToast('Video uploaded! Processing started.', 'success');
      startPolling(currentJobId);
    } else {
      const err = JSON.parse(xhr.responseText);
      showToast(`Upload failed: ${err.detail || 'Unknown error'}`, 'error');
      resetProgress();
    }
  });

  xhr.addEventListener('error', () => {
    showToast('Network error. Is the backend running?', 'error');
    resetProgress();
  });

  xhr.send(formData);
}

// ── Polling ──
function startPolling(jobId) {
  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/status/${jobId}`);
      const data = await res.json();

      if (data.status === 'processing') {
        setProgress(data.message || 'Processing...', 100);
        animateProgress();
      } else if (data.status === 'complete') {
        clearInterval(pollingInterval);
        setProgress('Complete!', 100);
        fetchResult(jobId);
      } else if (data.status === 'error') {
        clearInterval(pollingInterval);
        showToast(`Error: ${data.message}`, 'error');
        resetProgress();
      }
    } catch (e) {
      clearInterval(pollingInterval);
      showToast('Lost connection to backend.', 'error');
    }
  }, 3000);
}

// ── Fetch Result ──
async function fetchResult(jobId) {
  try {
    const res = await fetch(`${API_BASE}/result/${jobId}`);
    const data = await res.json();
    renderResults(data.result, data.filename);
  } catch (e) {
    showToast('Failed to fetch results.', 'error');
  }
}

// ── Render Results ──
function renderResults(result, filename) {
  uploadSection.style.display = 'none';
  resultsSection.style.display = 'block';

  const report = result.report;

  // Threat level
  const threatLevel = document.getElementById('threatLevel');
  threatLevel.textContent = report.threat_level;
  threatLevel.className = `threat-level ${report.threat_level}`;
  document.getElementById('threatStats').textContent =
    `${report.anomaly_frames} anomalous frames out of ${report.total_frames} total`;

  // Videos
  if (uploadedFileURL) {
    document.getElementById('originalVideo').src = uploadedFileURL;
  }
  document.getElementById('heatmapVideo').src = `http://localhost:8000${result.heatmap_video_url}`;

  // Timeline
  renderTimeline(report.frame_scores);

  // Metrics
  renderMetrics(report);

  showToast('Analysis complete!', 'success');
  progressWrapper.style.display = 'none';
}

// ── Timeline ──
function renderTimeline(frameScores) {
  const bar = document.getElementById('timelineBar');
  bar.innerHTML = '';
  if (!frameScores || frameScores.length === 0) return;

  // Downsample to max 200 segments for display
  const total = frameScores.length;
  const maxSegs = 200;
  const step = Math.max(1, Math.floor(total / maxSegs));

  for (let i = 0; i < total; i += step) {
    const seg = frameScores[i];
    const div = document.createElement('div');
    div.className = `timeline-segment ${seg.status}`;
    div.style.flex = `${step}`;
    div.title = `Frame ${seg.frame_idx} | Score: ${seg.score.toFixed(5)} | ${seg.status}`;
    bar.appendChild(div);
  }
}

// ── Metrics ──
function renderMetrics(report) {
  const grid = document.getElementById('metricsGrid');
  const metrics = [
    { label: 'Total Frames',      value: report.total_frames,      color: '' },
    { label: 'Normal Frames',     value: report.normal_frames,     color: '#22c55e' },
    { label: 'Suspicious Frames', value: report.suspicious_frames, color: '#eab308' },
    { label: 'Anomaly Frames',    value: report.anomaly_frames,    color: '#ef4444' },
    { label: 'Max Score',         value: report.max_score.toFixed(5), color: '' },
    { label: 'Avg Score',         value: report.avg_score.toFixed(5), color: '' },
  ];

  grid.innerHTML = metrics.map(m => `
    <div class="metric-card">
      <div class="metric-value" style="color:${m.color || 'var(--text)'}">${m.value}</div>
      <div class="metric-label">${m.label}</div>
    </div>
  `).join('');
}

// ── Helpers ──
function setProgress(label, pct) {
  progressLabel.textContent = label;
  progressBarFill.style.width = `${pct}%`;
  progressStatus.textContent = `${pct}%`;
}

function animateProgress() {
  let current = parseFloat(progressBarFill.style.width) || 0;
  if (current < 95) {
    current += 0.5;
    progressBarFill.style.width = `${current}%`;
    progressStatus.textContent = `${Math.round(current)}%`;
  }
}

function resetProgress() {
  progressWrapper.style.display = 'none';
  progressBarFill.style.width = '0%';
}

function resetUI() {
  resultsSection.style.display = 'none';
  uploadSection.style.display  = 'block';
  progressWrapper.style.display = 'none';
  progressBarFill.style.width   = '0%';
  fileInput.value = '';
  currentJobId = null;
  uploadedFileURL = null;
  if (pollingInterval) clearInterval(pollingInterval);
}

function showToast(msg, type = '') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  setTimeout(() => { toast.className = 'toast'; }, 4000);
}