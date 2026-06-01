/*
 * GSplat Pro Controller - WebSocket & Job Queue Enabled
 */

const API_BASE = '/api';

// State
let appState = {
    file: null,
    socket: null,
    selectedJobId: null,
    isTraining: false,
    isViewing: false,
    lastLogStr: ""
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
    modeValue: document.getElementById('mode-value'),
    
    queueList: document.getElementById('queue-list'),
    queueCount: document.getElementById('queue-count'),
    
    terminal: document.getElementById('terminal-output'),
    logCount: document.getElementById('log-count'),
    clearLogsBtn: document.getElementById('clear-logs-btn'),
    
    placeholderView: document.getElementById('placeholder-view'),
    viewerFrame: document.getElementById('viewer-frame'),
    
    checkpointsList: document.getElementById('checkpoints-list'),
    rendersGrid: document.getElementById('renders-grid')
};

// --- Initialization ---

function init() {
    setupEventListeners();
    initWebSocket();
    fetchInfo();
    log("System initialized. Ready.", "system");
}

async function fetchInfo() {
    try {
        const response = await fetch(`${API_BASE}/info`);
        if (response.ok) {
            const data = await response.json();
            if (data.has_existing_poses) {
                document.getElementById('skip-ace').checked = true;
                log("Detected existing camera poses on server. 'Skip Step 1' enabled by default.", "info");
            }
        }
    } catch (err) {
        console.error("Failed to fetch server info", err);
    }
}

function setupEventListeners() {
    // Upload
    dom.uploadBtn.addEventListener('click', () => dom.videoInput.click());
    dom.videoInput.addEventListener('change', handleFileUpload);
    
    // Actions
    dom.trainBtn.addEventListener('click', startTraining);
    dom.stopBtn.addEventListener('click', stopActiveTraining);
    dom.launchViewerBtn.addEventListener('click', launchViewer);
    
    // Utils
    dom.clearLogsBtn.addEventListener('click', () => {
        dom.terminal.innerHTML = '';
        updateLogCount(0);
    });

    // Tab Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const tabName = item.getAttribute('data-tab');
            switchTab(tabName);
        });
    });

    // Refresh Buttons
    document.getElementById('refresh-checkpoints-btn')?.addEventListener('click', loadCheckpoints);
    document.getElementById('refresh-renders-btn')?.addEventListener('click', loadRenders);
}

// --- WebSocket Management ---

function initWebSocket() {
    let wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let wsUrl = `${wsProto}//${window.location.host}/api/ws`;
    
    log(`Connecting to WebSocket: ${wsUrl}`, "system");
    
    appState.socket = new WebSocket(wsUrl);
    
    appState.socket.onopen = () => {
        log("WebSocket connected.", "success");
        // Clear terminal output initially
        dom.terminal.innerHTML = '<div class="log-line system">Connected to real-time stream.</div>';
    };
    
    appState.socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWebSocketEvent(msg);
    };
    
    appState.socket.onclose = () => {
        log("WebSocket disconnected. Retrying in 3 seconds...", "warning");
        setTimeout(initWebSocket, 3000);
    };
    
    appState.socket.onerror = (err) => {
        console.error("WebSocket error", err);
    };
}

function handleWebSocketEvent(msg) {
    switch (msg.event) {
        case "connected":
        case "queue.update":
            updateQueue(msg.queue);
            break;
            
        case "job.status":
            updateJobState(msg);
            break;
            
        case "job.log":
            if (appState.selectedJobId === msg.job_id) {
                appendLogLine(msg.line);
            }
            break;
            
        case "logs_replay":
            if (appState.selectedJobId === msg.job_id) {
                renderLogs(msg.logs);
            }
            break;
            
        case "job.complete":
            if (appState.selectedJobId === msg.job_id) {
                log("Job completed successfully!", "success");
                setTrainingState(false);
                dom.launchViewerBtn.classList.remove('btn-outline');
                dom.launchViewerBtn.classList.add('btn-primary');
            }
            break;
            
        case "job.error":
            if (appState.selectedJobId === msg.job_id) {
                log(`Job failed: ${msg.error}`, "error");
                setTrainingState(false);
            }
            break;
    }
}

// --- Queue UI updates ---

function updateQueue(queue) {
    if (!queue || queue.length === 0) {
        dom.queueList.innerHTML = `
            <div class="queue-empty">
                <p>No enqueued jobs. Submit a training config above to get started.</p>
            </div>
        `;
        dom.queueCount.textContent = "0 jobs";
        return;
    }
    
    dom.queueCount.textContent = `${queue.length} job(s)`;
    dom.queueList.innerHTML = '';
    
    queue.forEach(job => {
        const item = document.createElement('div');
        item.className = `queue-item ${appState.selectedJobId === job.job_id ? 'active' : ''}`;
        item.dataset.jobId = job.job_id;
        
        let badgeClass = `badge-status badge-${job.status}`;
        let statusText = job.status.toUpperCase();
        
        item.innerHTML = `
            <div class="queue-item-meta">
                <div class="queue-item-title">
                    <i class="fa-solid fa-file-video"></i> ${job.filename}
                    <span class="${badgeClass}">${statusText}</span>
                </div>
                <div class="queue-item-sub">
                    <span>ID: ${job.job_id.substring(0, 8)}...</span>
                    <span>Created: ${job.created_at}</span>
                    <span>Progress: ${job.progress}%</span>
                </div>
            </div>
            <div class="queue-item-actions">
                ${['queued', 'running', 'processing'].includes(job.status) ? 
                    `<button class="btn btn-sm btn-danger cancel-job-btn" data-job-id="${job.job_id}">
                        <i class="fa-solid fa-xmark"></i> Cancel
                     </button>` : ''
                }
            </div>
        `;
        
        // Item click selects that job
        item.addEventListener('click', (e) => {
            if (e.target.closest('.cancel-job-btn')) return;
            selectJob(job.job_id);
        });
        
        // Cancel click
        const cancelBtn = item.querySelector('.cancel-job-btn');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                cancelJob(job.job_id);
            });
        }
        
        dom.queueList.appendChild(item);
    });
    
    // Auto-select first enqueued/running job if none is selected
    if (!appState.selectedJobId && queue.length > 0) {
        selectJob(queue[0].job_id);
    }
}

function selectJob(jobId) {
    if (appState.selectedJobId === jobId && document.querySelector('.queue-item.active')) return;
    
    // Unsubscribe from previous if connected
    if (appState.selectedJobId && appState.socket && appState.socket.readyState === WebSocket.OPEN) {
        appState.socket.send(JSON.stringify({
            event: "unsubscribe",
            job_id: appState.selectedJobId
        }));
    }
    
    appState.selectedJobId = jobId;
    
    // Highlight in UI
    document.querySelectorAll('.queue-item').forEach(item => {
        if (item.dataset.jobId === jobId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Reset terminal output
    dom.terminal.innerHTML = '<div class="log-line system">Loading logs...</div>';
    
    // Subscribe
    if (appState.socket && appState.socket.readyState === WebSocket.OPEN) {
        appState.socket.send(JSON.stringify({
            event: "subscribe",
            job_id: jobId
        }));
    }
}

function updateJobState(job) {
    if (appState.selectedJobId !== job.job_id) return;
    
    const statusMap = {
        'queued': 'Queued',
        'running': 'Running',
        'processing': 'Processing',
        'completed': 'Complete',
        'failed': 'Failed',
        'cancelled': 'Cancelled'
    };
    
    dom.statusValue.textContent = statusMap[job.status] || job.status.toUpperCase();
    
    const configMode = job.config ? job.config.mode : 'ace_zero';
    dom.modeValue.textContent = configMode === 'ace_zero' ? 'ACE-Zero' : (configMode === 'simple' ? 'Simple' : 'Chunked');
    
    dom.statusCard.className = `kpi-card ${['running', 'processing'].includes(job.status) ? 'active-pulse' : ''}`;
    
    dom.progressValue.textContent = `${job.progress}%`;
    dom.progressBarFill.style.width = `${job.progress}%`;
    
    if (['running', 'processing'].includes(job.status)) {
        setTrainingState(true);
    } else {
        setTrainingState(false);
        if (job.status === 'completed') {
            dom.launchViewerBtn.classList.remove('btn-outline');
            dom.launchViewerBtn.classList.add('btn-primary');
        }
    }
}

// --- Handlers & API Operations ---

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
    
    const config = {
        mode: document.getElementById('mode-select').value,
        quality_mode: document.getElementById('quality-mode').value,
        max_steps: parseInt(document.getElementById('max-steps').value) || 7000,
        fps: parseFloat(document.getElementById('fps').value) || 2.0,
        data_factor: parseInt(document.getElementById('data-factor').value) || 2,
        depth_model: document.getElementById('depth-model').value || 'depth_anything',
        
        sh_degree: parseInt(document.getElementById('sh-degree').value) || 3,
        means_lr: parseFloat(document.getElementById('means-lr').value) || 0.00016,
        ssim_lambda: parseFloat(document.getElementById('ssim-lambda').value) || 0.2,
        random_bkgd: document.getElementById('random-bkgd').checked,
        pose_opt: document.getElementById('pose-opt').checked,
        app_opt: document.getElementById('app-opt').checked,
        skip_ace: document.getElementById('skip-ace').checked
    };

    try {
        const response = await fetch(`${API_BASE}/jobs/submit`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        if (!response.ok) throw new Error("Failed to submit job to queue");
        
        const data = await response.json();
        log(`Job submitted. Job ID: ${data.job_id.substring(0, 8)}...`, "success");
        selectJob(data.job_id);
        
    } catch (err) {
        log(`Submission failed: ${err.message}`, "error");
    }
}

async function stopActiveTraining() {
    if (!appState.selectedJobId) return;
    await cancelJob(appState.selectedJobId);
}

async function cancelJob(jobId) {
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error("Failed to cancel job");
        log(`Cancellation request sent for job ${jobId.substring(0, 8)}...`, "warning");
    } catch (err) {
        log(`Error cancelling: ${err.message}`, "error");
    }
}

async function launchViewer() {
    try {
        log("Launching immersive viewer...", "info");
        const response = await fetch(`${API_BASE}/view`, { method: 'POST' });
        
        if (!response.ok) throw new Error("Viewer launch failed");
        
        const data = await response.json();
        const url = data.url; 
        
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

// --- Terminal Log Rendering Helpers ---

function appendLogLine(line) {
    const div = document.createElement('div');
    div.className = 'log-line';
    div.textContent = `> ${line}`;
    
    if (line.toLowerCase().includes('error')) div.classList.add('error');
    if (line.includes('warning') || line.includes('[Warning]')) div.classList.add('info'); 
    if (line.includes('SUCCESS') || line.includes('completed successfully')) div.classList.add('success');
    
    dom.terminal.appendChild(div);
    dom.terminal.scrollTop = dom.terminal.scrollHeight;
    
    updateLogCount(dom.terminal.querySelectorAll('.log-line').length);
}

function renderLogs(logs) {
    dom.terminal.innerHTML = ''; 
    if (!logs || logs.length === 0) {
        dom.terminal.innerHTML = '<div class="log-line system">Waiting for process logs...</div>';
        updateLogCount(0);
        return;
    }
    
    logs.forEach(line => {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.textContent = `> ${line}`;
        
        if (line.toLowerCase().includes('error')) div.classList.add('error');
        if (line.includes('warning') || line.includes('[Warning]')) div.classList.add('info'); 
        if (line.includes('SUCCESS') || line.includes('completed successfully')) div.classList.add('success');
        
        dom.terminal.appendChild(div);
    });
    
    dom.terminal.scrollTop = dom.terminal.scrollHeight;
    updateLogCount(logs.length);
}

function updateLogCount(count) {
    dom.logCount.textContent = `${count} lines`;
}

function setLoading(isLoading, text) {
    if (isLoading) {
        dom.uploadBtn.disabled = true;
        dom.uploadBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${text}`;
    } else {
        dom.uploadBtn.disabled = false;
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
    const div = document.createElement('div');
    div.className = `log-line ${type}`;
    div.textContent = `[CLIENT] ${msg}`;
    dom.terminal.appendChild(div);
    dom.terminal.scrollTop = dom.terminal.scrollHeight;
    updateLogCount(dom.terminal.querySelectorAll('.log-line').length);
}

function switchTab(tabName) {
    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('data-tab') === tabName) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // Update active section
    document.querySelectorAll('.tab-section').forEach(section => {
        if (section.id === `tab-${tabName}`) {
            section.classList.remove('hidden');
        } else {
            section.classList.add('hidden');
        }
    });

    // Update breadcrumb
    const activityName = document.querySelector('.activity-name');
    if (activityName) {
        activityName.textContent = tabName.charAt(0).toUpperCase() + tabName.slice(1);
    }

    // Perform tab-specific fetches
    if (tabName === 'checkpoints') {
        loadCheckpoints();
    } else if (tabName === 'renders') {
        loadRenders();
    }
}

async function loadCheckpoints() {
    if (!dom.checkpointsList) return;
    
    dom.checkpointsList.innerHTML = '<tr><td colspan="4" class="text-center"><i class="fa-solid fa-spinner fa-spin"></i> Fetching checkpoints...</td></tr>';
    
    try {
        const response = await fetch(`${API_BASE}/checkpoints`);
        if (!response.ok) throw new Error("Failed to fetch checkpoints");
        
        const data = await response.json();
        const ckpts = data.checkpoints || [];
        
        if (ckpts.length === 0) {
            dom.checkpointsList.innerHTML = '<tr><td colspan="4" class="text-center">No checkpoints found. Start training to save checkpoints.</td></tr>';
            return;
        }
        
        dom.checkpointsList.innerHTML = '';
        ckpts.forEach(ckpt => {
            const tr = document.createElement('tr');
            
            // Name cell
            const nameTd = document.createElement('td');
            nameTd.innerHTML = `<i class="fa-solid fa-file-invoice"></i> <strong>${ckpt.name}</strong>`;
            
            // Size cell
            const sizeTd = document.createElement('td');
            sizeTd.textContent = `${ckpt.size_mb} MB`;
            
            // Path cell
            const pathTd = document.createElement('td');
            pathTd.innerHTML = `<code style="font-size:0.8rem; color:var(--text-secondary);">${ckpt.id}</code>`;
            
            // Actions cell
            const actionsTd = document.createElement('td');
            actionsTd.style.textAlign = 'right';
            
            const viewBtn = document.createElement('button');
            viewBtn.className = 'btn btn-sm btn-primary';
            viewBtn.innerHTML = '<i class="fa-solid fa-eye"></i> View in 3D';
            viewBtn.addEventListener('click', () => viewCheckpoint(ckpt.path));
            
            actionsTd.appendChild(viewBtn);
            
            tr.appendChild(nameTd);
            tr.appendChild(sizeTd);
            tr.appendChild(pathTd);
            tr.appendChild(actionsTd);
            
            dom.checkpointsList.appendChild(tr);
        });
    } catch (err) {
        dom.checkpointsList.innerHTML = `<tr><td colspan="4" class="text-center" style="color:var(--accent-danger);">Error loading checkpoints: ${err.message}</td></tr>`;
    }
}

async function viewCheckpoint(ckptPath) {
    try {
        log(`Launching viewer for checkpoint: ${ckptPath}`, "info");
        const response = await fetch(`${API_BASE}/view`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ ckpt_paths: [ckptPath] })
        });
        
        if (!response.ok) throw new Error("Viewer launch failed");
        
        const data = await response.json();
        
        // Show success and switch back to Dashboard tab to display the 3D Viewport
        log(`Viewer active at ${data.url}`, "success");
        switchTab('dashboard');
        
        // Load in iframe
        setTimeout(() => {
            dom.viewerFrame.src = data.url;
            dom.placeholderView.classList.add('hidden');
            dom.viewerFrame.classList.remove('hidden');
        }, 1000);
    } catch (err) {
        log(`Viewer error: ${err.message}`, "error");
    }
}

async function loadRenders() {
    if (!dom.rendersGrid) return;
    
    dom.rendersGrid.innerHTML = '<div class="text-center" style="grid-column: 1 / -1; padding: 32px;"><i class="fa-solid fa-spinner fa-spin"></i> Fetching renders...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/renders`);
        if (!response.ok) throw new Error("Failed to fetch renders");
        
        const data = await response.json();
        const renders = data.renders || [];
        
        if (renders.length === 0) {
            dom.rendersGrid.innerHTML = '<div class="text-center" style="grid-column: 1 / -1; padding: 32px; color:var(--text-secondary);">No renders available yet. Start training to generate preview renders.</div>';
            return;
        }
        
        dom.rendersGrid.innerHTML = '';
        renders.forEach(render => {
            const card = document.createElement('div');
            card.className = 'render-card';
            
            card.innerHTML = `
                <div class="render-img-wrapper">
                    <img src="${render.url}" alt="${render.name}" onerror="this.src='placeholder.png';">
                </div>
                <div class="render-card-body">
                    <div class="render-card-title">${render.name}</div>
                    <div class="render-card-actions">
                        <a href="${render.url}" target="_blank" class="btn btn-sm btn-outline">
                            <i class="fa-solid fa-maximize"></i> Open Fullsize
                        </a>
                    </div>
                </div>
            `;
            
            dom.rendersGrid.appendChild(card);
        });
    } catch (err) {
        dom.rendersGrid.innerHTML = `<div class="text-center" style="grid-column: 1 / -1; padding: 32px; color:var(--accent-danger);">Error loading renders: ${err.message}</div>`;
    }
}

// Start controller
document.addEventListener('DOMContentLoaded', init);
