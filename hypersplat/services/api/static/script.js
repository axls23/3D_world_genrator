/*
 * GSplat Pro Controller
 */

const API_BASE = '/api';

// State
let appState = {
    file: null,
    isTraining: false,
    isViewing: false,
    logs: [],
    lastLogCount: 0,
    pollingInterval: null
};

// Elements
const dom = {
    uploadBtn: document.getElementById('upload-btn'),
    videoInput: document.getElementById('video-upload'),
    trainBtn: document.getElementById('train-btn'),
    stopBtn: document.getElementById('stop-btn'),
    launchViewerBtn: document.getElementById('launch-viewer-btn'),
    
    statusCard: document.getElementById('status-card'),
    statusValue: document.getElementById('status-value'),
    progressValue: document.getElementById('progress-value'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    
    terminal: document.getElementById('terminal-output'),
    logCount: document.getElementById('log-count'),
    clearLogsBtn: document.getElementById('clear-logs-btn'),
    
    placeholderView: document.getElementById('placeholder-view'),
    viewerFrame: document.getElementById('viewer-frame')
};

// --- Initialization ---

function init() {
    setupEventListeners();
    startPolling();
    log("System initialized. Ready.", "system");
}

function setupEventListeners() {
    // Upload
    dom.uploadBtn.addEventListener('click', () => dom.videoInput.click());
    dom.videoInput.addEventListener('change', handleFileUpload);
    
    // Actions
    dom.trainBtn.addEventListener('click', startTraining);
    dom.stopBtn.addEventListener('click', stopTraining);
    dom.launchViewerBtn.addEventListener('click', launchViewer);
    
    // Utils
    dom.clearLogsBtn.addEventListener('click', () => {
        dom.terminal.innerHTML = '';
        appState.lastLogCount = 0; // Reset count
        updateLogCount();
    });
}

// --- Handlers ---

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    setLoading(true, "Uploading...");
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("Upload failed");
        
        const data = await response.json();
        appState.file = data;
        log(`Video uploaded: ${data.filename} (${data.size_mb} MB)`, "info");
        dom.trainBtn.disabled = false;
        
        // Update header action text
        dom.uploadBtn.innerHTML = `<i class="fa-solid fa-check"></i> ${data.filename}`;
        dom.uploadBtn.classList.replace('btn-secondary', 'btn-outline');
        
    } catch (err) {
        log(`Error uploading: ${err.message}`, "error");
    } finally {
        setLoading(false);
    }
}

async function startTraining() {
    if (!appState.file && !confirm("No new file uploaded. Use existing file on server?")) return;
    
    // Gather Settings
    const config = {
        mode: document.getElementById('mode-select').value,
        quality_mode: document.getElementById('quality-mode').value,
        max_steps: parseInt(document.getElementById('max-steps').value) || 7000,
        
        // Advanced Hyperparams
        sh_degree: parseInt(document.getElementById('sh-degree').value) || 3,
        means_lr: parseFloat(document.getElementById('means-lr').value) || 0.00016,
        ssim_lambda: parseFloat(document.getElementById('ssim-lambda').value) || 0.2,
        random_bkgd: document.getElementById('random-bkgd').checked,
        pose_opt: document.getElementById('pose-opt').checked,
        app_opt: document.getElementById('app-opt').checked
    };

    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        if (!response.ok) throw new Error("Failed to start training");
        
        const data = await response.json();
        log("Training started via ACE-Zero Pipeline", "success");
        setTrainingState(true);
        
    } catch (err) {
        log(`Start failed: ${err.message}`, "error");
    }
}

async function stopTraining() {
    try {
        await fetch(`${API_BASE}/stop`, { method: 'POST' });
        log("Stop command sent", "warning");
    } catch (err) {
        log(`Error stopping: ${err.message}`, "error");
    }
}

async function launchViewer() {
    try {
        log("Launching immersive viewer...", "info");
        const response = await fetch(`${API_BASE}/view`, { method: 'POST' });
        
        if (!response.ok) throw new Error("Viewer launch failed");
        
        const data = await response.json();
        const url = data.url; // e.g., http://localhost:8092
        
        // Since iframe is on same domain or we want to mix content?
        // Note: demo_server.py usually runs on 8080, viewer on 8092.
        // We might need to proxy or just use the full URL.
        
        // Force the iframe to reload by adding timestamp
        // NOTE: The demo_server code returns 'url' which is typically http://localhost:8092
        // We set this directly to the src. 
        // Using a slight delay to allow process to spin up
        
        setTimeout(() => {
            dom.viewerFrame.src = url;
            dom.placeholderView.classList.add('hidden');
            dom.viewerFrame.classList.remove('hidden');
            log(`Viewer active at ${url}`, "success");
        }, 1000);
        
    } catch (err) {
        log(`Viewer error: ${err.message}`, "error");
    }
}

// --- Polling & Updates ---

function startPolling() {
    if (appState.pollingInterval) clearInterval(appState.pollingInterval);
    appState.pollingInterval = setInterval(pollBackend, 1000);
}

async function pollBackend() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) return;
        
        const data = await response.json();
        updateUI(data);
        
    } catch (err) {
        console.error("Poll error", err);
    }
}

function updateUI(data) {
    // 1. Status Text
    const statusMap = {
        'idle': 'Ready',
        'training_running': 'Training Active',
        'training_complete': 'Complete',
        'error': 'Error',
        'viewing': 'Viewing'
    };
    
    dom.statusValue.textContent = statusMap[data.status] || data.status;
    
    // 2. Status Color/Card style
    dom.statusCard.className = `kpi-card ${data.status === 'training_running' ? 'active-pulse' : ''}`;
    
    // 3. Progress
    const prog = data.progress || 0;
    dom.progressValue.textContent = `${prog}%`;
    dom.progressBarFill.style.width = `${prog}%`;
    
    // 4. Buttons State
    if (data.status === 'training_running') {
        setTrainingState(true);
    } else if (data.status === 'training_complete') {
        setTrainingState(false);
        dom.launchViewerBtn.classList.remove('btn-outline');
        dom.launchViewerBtn.classList.add('btn-primary');
    } else {
        setTrainingState(false);
    }
    
    // 5. Logs
    if (data.logs && data.logs.length > 0) {
        // Simple diff: if new logs length > old logs length
        // Note: The API returns last 50 logs. 
        // We should append new ones. 
        // Setup simple dedup based on content could be tricky if repeated lines.
        // For now, let's just render the batch if it changed significantly or just render all
        // since it's only 50 lines.
        
        // Clearing logic checks
        const currentLogStr = JSON.stringify(data.logs);
        if (appState.lastLogStr !== currentLogStr) {
            renderLogs(data.logs);
            appState.lastLogStr = currentLogStr;
        }
    }

    // 6. Check for Viewer URL from training (Live Training Viewer)
    if (data.training_viewer_url && !appState.isViewing) {
        // If we want to show the training viewer automatically?
        // Let's just log it for now
        // log(`Training Viewer available: ${data.training_viewer_url}`, 'info');
    }
}

function renderLogs(logs) {
    // Don't clear manual history, just append batch? 
    // Actually, backend returns last 50. Let's just replace the view to stay synced 
    // or we might miss lines in between polls. 
    // For a robust terminal, we'd need a seek/cursor API. 
    // Given the constraints, let's replace innerHTML with the latest snapshot 
    // PLUS any local history we want to keep? 
    // Let's just show the window returned by server for now.
    
    dom.terminal.innerHTML = ''; 
    
    logs.forEach(line => {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.textContent = `> ${line}`;
        
        // Simple heuristic for colors
        if (line.toLowerCase().includes('error')) div.classList.add('error');
        if (line.includes('warning')) div.classList.add('info'); // yellowish
        if (line.includes('SUCCESS') || line.includes('Complete')) div.classList.add('success');
        
        dom.terminal.appendChild(div);
    });
    
    // Auto scroll to bottom
    dom.terminal.scrollTop = dom.terminal.scrollHeight;
    
    // Update count
    dom.logCount.textContent = `${logs.length} lines`;
}

// --- Helpers ---

function setLoading(isLoading, text) {
    if (isLoading) {
        dom.uploadBtn.disabled = true;
        dom.uploadBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${text}`;
    } else {
        dom.uploadBtn.disabled = false;
        // Text resets in handleFileUpload success path
    }
}

function setTrainingState(isRunning) {
    appState.isTraining = isRunning;
    
    if (isRunning) {
        dom.trainBtn.classList.add('hidden');
        dom.stopBtn.classList.remove('hidden');
    } else {
        dom.trainBtn.classList.remove('hidden');
        dom.stopBtn.classList.add('hidden');
    }
}

function log(msg, type = "system") {
    // Client-side only log injection
    const div = document.createElement('div');
    div.className = `log-line ${type}`;
    div.textContent = `[CLIENT] ${msg}`;
    dom.terminal.appendChild(div);
    dom.terminal.scrollTop = dom.terminal.scrollHeight;
}

// Start
document.addEventListener('DOMContentLoaded', init);
