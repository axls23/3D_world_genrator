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
    lastLogStr: "",
    mode: 'ace_zero',
    serverVideo: null,
    hasExistingPoses: false
};

// Entry points. Replaced by /api/info's pipeline_modes when the server answers,
// so the buttons always match what the backend dispatches.
let pipelineModes = {
    ace_zero: { label: 'ACE-Zero', level: 2, input: 'video', recommended: true,
                description: 'Video → AI camera poses → 3DGS. Unset params are tuned per scene.' },
    simple:   { label: 'Simple', level: 1, input: 'colmap', recommended: false,
                description: 'Train directly on an existing COLMAP dataset (no video, no pose step).' },
    chunked:  { label: 'Chunked', level: 3, input: 'video', recommended: false,
                description: 'Video → COLMAP-forced run. Experimental: chunking is not implemented yet.' }
};

// Per-mode wording for the primary action.
const MODE_ACTION = {
    ace_zero: 'Generate 3D Scene',
    simple: 'Train on Dataset',
    chunked: 'Run Chunked Pipeline'
};

// Stage toggles: tri-state buttons cycle auto -> on -> off. `value` maps each
// state to what the API receives (undefined = not sent = manager decides).
const TRI_STAGES = {
    skip_ace: { labels: { auto: 'Auto', on: 'Reuse', off: 'Re-estimate' },
                value: { on: true, off: false } },
    prune:    { labels: { auto: 'Auto', on: 'On', off: 'Off' },
                value: { on: true, off: false } }
};
const TRI_ORDER = ['auto', 'on', 'off'];

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
    modeSubtext: document.getElementById('mode-subtext'),

    modeSwitch: document.getElementById('mode-switch'),
    modeDescription: document.getElementById('mode-description'),
    videoRow: document.getElementById('video-input-row'),
    videoName: document.getElementById('video-name'),
    chooseVideoBtn: document.getElementById('choose-video-btn'),
    colmapRow: document.getElementById('colmap-input-row'),
    colmapInput: document.getElementById('colmap-input'),
    stageToggles: document.getElementById('stage-toggles'),
    trainBtnLabel: document.getElementById('train-btn-label'),
    trainHint: document.getElementById('train-hint'),
    autoCount: document.getElementById('auto-count'),
    
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
    renderModeButtons();
    selectMode(appState.mode);
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
            if (data.pipeline_modes) {
                pipelineModes = data.pipeline_modes;
                if (!pipelineModes[appState.mode]) appState.mode = data.default_mode || Object.keys(pipelineModes)[0];
                renderModeButtons();
                selectMode(appState.mode);
            }
            appState.hasExistingPoses = !!data.has_existing_poses;
            if (data.current_video) {
                appState.serverVideo = data.current_video;
                updateEntryInput();
            }
            if (appState.hasExistingPoses) {
                log("Existing camera poses found on server. 'Poses: Auto' will reuse them.", "info");
            }
        }
    } catch (err) {
        console.error("Failed to fetch server info", err);
    }
}

function setupEventListeners() {
    // Upload
    dom.uploadBtn.addEventListener('click', () => dom.videoInput.click());
    dom.chooseVideoBtn.addEventListener('click', () => dom.videoInput.click());
    dom.videoInput.addEventListener('change', handleFileUpload);
    dom.colmapInput.addEventListener('input', updateTrainAvailability);

    // Entry point + stage toggles
    dom.stageToggles.querySelectorAll('.stage-btn').forEach(btn => {
        btn.addEventListener('click', () => toggleStage(btn));
    });
    document.querySelectorAll('#tab-settings [data-key]').forEach(el => {
        el.addEventListener('input', updateAutoCount);
        el.addEventListener('change', updateAutoCount);
    });
    
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

// --- Entry Point (mode) Controls ---

function renderModeButtons() {
    dom.modeSwitch.innerHTML = '';
    Object.entries(pipelineModes).forEach(([key, mode]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mode-btn';
        btn.dataset.mode = key;
        btn.setAttribute('role', 'radio');
        btn.setAttribute('aria-checked', String(key === appState.mode));

        const title = document.createElement('span');
        title.className = 'mode-title';
        title.textContent = mode.label;
        if (mode.recommended) {
            const tag = document.createElement('span');
            tag.className = 'mode-tag';
            tag.textContent = 'Recommended';
            title.appendChild(tag);
        } else if (/experimental/i.test(mode.description || '')) {
            const tag = document.createElement('span');
            tag.className = 'mode-tag warn';
            tag.textContent = 'Experimental';
            title.appendChild(tag);
        }

        const meta = document.createElement('span');
        meta.className = 'mode-meta';
        meta.textContent = `Level ${mode.level} · ${mode.input === 'colmap' ? 'COLMAP dataset' : 'Video'} input`;

        btn.append(title, meta);
        btn.addEventListener('click', () => selectMode(key));
        dom.modeSwitch.appendChild(btn);
    });
}

function selectMode(key) {
    if (appState.isTraining) return;
    appState.mode = key;
    const mode = pipelineModes[key];

    dom.modeSwitch.querySelectorAll('.mode-btn').forEach(btn => {
        btn.setAttribute('aria-checked', String(btn.dataset.mode === key));
    });
    dom.modeDescription.textContent = mode.description || '';
    dom.trainBtnLabel.textContent = MODE_ACTION[key] || 'Start';
    setModeKpi(key);

    // Pipeline stages and most tunables only exist in the manager (video modes).
    const isVideo = mode.input !== 'colmap';
    dom.stageToggles.querySelectorAll('.stage-btn').forEach(btn => { btn.disabled = !isVideo; });
    document.querySelectorAll('#tab-settings [data-modes]').forEach(group => {
        const applies = group.dataset.modes.split(' ').includes(key);
        group.classList.toggle('is-disabled', !applies);
        group.querySelectorAll('input, select').forEach(el => { el.disabled = !applies; });
    });

    updateEntryInput();
    updateAutoCount();
}

function setModeKpi(key) {
    const mode = pipelineModes[key] || { label: key, level: '?' };
    dom.modeValue.textContent = mode.label;
    dom.modeSubtext.textContent = `Level ${mode.level} Pipeline`;
}

function updateEntryInput() {
    const isColmap = pipelineModes[appState.mode].input === 'colmap';
    dom.videoRow.classList.toggle('hidden', isColmap);
    dom.colmapRow.classList.toggle('hidden', !isColmap);
    dom.uploadBtn.classList.toggle('hidden', isColmap);

    const name = appState.file ? appState.file.filename : appState.serverVideo;
    dom.videoName.textContent = name ? name : 'No video selected';
    updateTrainAvailability();
}

function updateTrainAvailability() {
    const isColmap = pipelineModes[appState.mode].input === 'colmap';
    let ready, hint;
    if (isColmap) {
        ready = dom.colmapInput.value.trim().length > 0;
        hint = ready ? 'Trains on the dataset path on the server' : 'Enter a COLMAP dataset path';
    } else {
        ready = !!(appState.file || appState.serverVideo);
        hint = ready ? `Input: ${appState.file ? appState.file.filename : appState.serverVideo}` : 'Choose a video to start';
    }
    dom.trainBtn.disabled = !ready;
    dom.trainHint.textContent = hint;
}

function toggleStage(btn) {
    const key = btn.dataset.stage;
    const tri = TRI_STAGES[key];
    if (tri) {
        const next = TRI_ORDER[(TRI_ORDER.indexOf(btn.dataset.state) + 1) % TRI_ORDER.length];
        btn.dataset.state = next;
        btn.querySelector('b').textContent = tri.labels[next];
    } else {
        btn.setAttribute('aria-pressed', String(btn.getAttribute('aria-pressed') !== 'true'));
    }
    updateAutoCount();
}

// Only values the user set are included; everything else stays "auto" and the
// manager derives it per scene (see hypersplat.pipeline.wrapper).
function collectConfig() {
    const config = { mode: appState.mode };
    const isColmap = pipelineModes[appState.mode].input === 'colmap';
    if (isColmap) config.colmap_input = dom.colmapInput.value.trim();

    document.querySelectorAll('#tab-settings [data-key]').forEach(el => {
        if (el.disabled) return;
        const key = el.dataset.key;
        if (el.type === 'checkbox') {
            if (el.checked) config[key] = true;
            return;
        }
        const raw = el.value.trim();
        if (raw === '') return;
        const type = el.dataset.type;
        if (type === 'int') config[key] = parseInt(raw, 10);
        else if (type === 'float') config[key] = parseFloat(raw);
        else if (type === 'bool') config[key] = raw === 'true';
        else config[key] = raw;
    });

    if (!isColmap) {
        dom.stageToggles.querySelectorAll('.stage-btn').forEach(btn => {
            const key = btn.dataset.stage;
            const tri = TRI_STAGES[key];
            if (tri) {
                const value = tri.value[btn.dataset.state];
                if (value !== undefined) config[key] = value;
            } else if (btn.getAttribute('aria-pressed') === 'true') {
                config[key] = true;
            }
        });
    }
    return config;
}

function updateAutoCount() {
    const pinned = Object.keys(collectConfig()).filter(k => !['mode', 'colmap_input'].includes(k));
    dom.autoCount.textContent = pinned.length === 0 ? 'All params auto' : `${pinned.length} pinned · rest auto`;
    dom.autoCount.title = pinned.length ? `Pinned: ${pinned.join(', ')}` : 'Params left unset are tuned per scene by the manager';
    document.querySelectorAll('#tab-settings [data-key]').forEach(el => {
        el.classList.toggle('is-set', el.type !== 'checkbox' && el.value.trim() !== '' && !el.disabled);
    });
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
                    <i class="fa-solid ${job.mode === 'simple' ? 'fa-folder-open' : 'fa-file-video'}"></i> ${job.filename}
                    <span class="badge">${(pipelineModes[job.mode] || {}).label || job.mode || ''}</span>
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
    
    setModeKpi(job.mode || 'ace_zero');
    
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
        
        dom.uploadBtn.innerHTML = `<i class="fa-solid fa-check"></i> ${data.filename}`;
        dom.uploadBtn.classList.replace('btn-secondary', 'btn-outline');
        
    } catch (err) {
        log(`Error uploading: ${err.message}`, "error");
        dom.uploadBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Upload Video';
    } finally {
        setLoading(false);
        updateEntryInput();
    }
}

async function startTraining() {
    const config = collectConfig();
    dom.trainBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/jobs/submit`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            // Surface the server's validation message (e.g. "fps must be a positive number")
            const body = await response.json().catch(() => ({}));
            throw new Error(typeof body.detail === 'string' ? body.detail : "Failed to submit job to queue");
        }
        
        const data = await response.json();
        const pinned = Object.keys(config).filter(k => !['mode', 'colmap_input'].includes(k));
        log(`${pipelineModes[config.mode].label} job submitted (${data.job_id.substring(0, 8)}...). ` +
            (pinned.length ? `Pinned: ${pinned.join(', ')}; rest auto.` : 'All params auto.'), "success");
        selectJob(data.job_id);
        
    } catch (err) {
        log(`Submission failed: ${err.message}`, "error");
    } finally {
        updateTrainAvailability();
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
        dom.trainHint.textContent = 'Run in progress';
    } else {
        dom.trainBtn.classList.remove('hidden');
        dom.stopBtn.classList.add('hidden');
        updateTrainAvailability();
    }
    // Mode can't change mid-run; the KPI shows the running job's mode.
    dom.modeSwitch.querySelectorAll('.mode-btn').forEach(btn => { btn.disabled = isRunning; });
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
