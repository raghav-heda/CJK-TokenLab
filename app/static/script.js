const languages = {};
let allMetrics = {};
let radarChart, baseChart, propChart, ablationChart;

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initGlobalData();
    initEventListeners();
});

// Navigation handling
function initNavigation() {
    const navBtns = document.querySelectorAll(".nav-btn");
    navBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            // Remove active from all
            navBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".page-section").forEach(s => s.classList.remove("active"));
            
            // Add active to clicked
            btn.classList.add("active");
            document.getElementById(btn.dataset.target).classList.add("active");
        });
    });
}

// Global data fetching
async function initGlobalData() {
    try {
        const res = await fetch('/api/config');
        if(!res.ok) throw new Error("Failed to connect to backend API.");
        const data = await res.json();
        
        Object.assign(languages, data.languages);
        
        const selects = [
            document.getElementById("langSelect"),
            document.getElementById("metricsLangSelect"),
            document.getElementById("evalLangSelect")
        ];
        
        for (const [code, name] of Object.entries(languages)) {
            selects.forEach(s => s.add(new Option(name, code)));
        }
        
        // Initial fetches
        if(selects[0].value) {
            loadStatus(selects[0].value);
        }
        loadMetrics();
        loadEvalCharts();
        loadAblation();
    } catch(err) {
        showToast("Error connecting to backend API: " + err.message, "error");
    }
}

// Event Listeners
function initEventListeners() {
    // Tokenizer
    document.getElementById("tokenizeBtn").addEventListener("click", handleTokenize);
    
    // Character Analyzer
    document.getElementById("analyzeCharBtn").addEventListener("click", handleCharacterAnalyze);
    
    // Selectors
    document.getElementById("metricsLangSelect").addEventListener("change", renderMetricsDashboard);
    document.getElementById("evalLangSelect").addEventListener("change", loadEvalCharts);
}

// Tokenize Feature
async function handleTokenize() {
    const text = document.getElementById("textInput").value;
    const lang_code = document.getElementById("langSelect").value;
    const btn = document.getElementById("tokenizeBtn");
    const btnText = btn.querySelector(".btn-text");
    const spinner = btn.querySelector(".spinner");
    
    if(!text.trim()) {
        showToast("Please enter some text to tokenize.", "error");
        return;
    }
    
    btn.disabled = true;
    btnText.classList.add("hidden");
    spinner.classList.remove("hidden");
    
    try {
        const res = await fetch('/api/tokenize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text, lang_code})
        });
        
        if(!res.ok) throw new Error("Tokenization failed.");
        const data = await res.json();
        
        const bLen = data.baseline.length;
        const pLen = data.proposed.length;
        
        document.getElementById("baseTokensCount").innerText = `${bLen || 0} tokens`;
        document.getElementById("baseTokenIds").innerHTML = renderTokenChips(data.baseline);
        
        document.getElementById("propTokensCount").innerText = `${pLen || 0} tokens`;
        document.getElementById("propTokenIds").innerHTML = renderTokenChips(data.proposed);
        
        const deltaEl = document.getElementById("propDelta");
        if(bLen && pLen) {
            const diff = pLen - bLen;
            const pct = Math.abs((diff / bLen) * 100).toFixed(1);
            if(diff < 0) {
                deltaEl.innerText = `${Math.abs(diff)} fewer tokens (${pct}% tighter)`;
                deltaEl.className = "val trend-up";
            } else if (diff > 0) {
                deltaEl.innerText = `${diff} more tokens (${pct}% looser)`;
                deltaEl.className = "val trend-down";
            } else {
                deltaEl.innerText = "Identical count";
                deltaEl.className = "val";
            }
        }
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        btn.disabled = false;
        btnText.classList.remove("hidden");
        spinner.classList.add("hidden");
    }
}

function renderTokenChips(tokens) {
    if(!tokens || tokens.length === 0) return "<div class='empty-state'>No tokens generated.</div>";
    return tokens.map(t => {
        const text = t.text || "?";
        const id = t.id !== undefined ? t.id : t;
        // Check for UNK
        const isUnk = (text === "[UNK]" || text === "<unk>");
        const styleClass = isUnk ? "border-color: var(--error);" : "";
        const idStyle = isUnk ? "color: var(--error);" : "";
        
        return `<div class="token-badge" style="${styleClass}">
            <span class="token-text">${text}</span>
            <span class="token-id" style="${idStyle}">${id}</span>
        </div>`;
    }).join("");
}

// Character Anatomy
async function handleCharacterAnalyze() {
    const chars = document.getElementById("charInput").value;
    const btn = document.getElementById("analyzeCharBtn");
    const btnText = btn.querySelector(".btn-text");
    const spinner = btn.querySelector(".spinner");
    
    if(!chars.trim()) return;
    
    btn.disabled = true;
    btnText.classList.add("hidden");
    spinner.classList.remove("hidden");
    
    try {
        const res = await fetch(`/api/analyze_chars?chars=${encodeURIComponent(chars)}`);
        if(!res.ok) throw new Error("Failed to analyze characters.");
        const data = await res.json();
        
        const grid = document.getElementById("charGrid");
        grid.innerHTML = "";
        
        if(data.length === 0) {
            grid.innerHTML = "<div class='empty-state' style='grid-column: 1/-1;'>No valid characters found.</div>";
        }
        
        data.forEach(d => {
            let html = `<div class="char-card glass-panel">
                <div class="char-script">${d.script}</div>
                <div class="char-main">${d.char}</div>
                <div class="char-details">`;
            if (d.decomposition) html += `<div class="char-detail-row"><span class="char-detail-label">IDS</span><span class="char-detail-val">${d.decomposition}</span></div>`;
            if (d.hiragana) html += `<div class="char-detail-row"><span class="char-detail-label">Kana</span><span class="char-detail-val">${d.hiragana}</span></div>`;
            if (d.jamo) html += `<div class="char-detail-row"><span class="char-detail-label">Jamo</span><span class="char-detail-val">${d.jamo}</span></div>`;
            html += `</div></div>`;
            grid.innerHTML += html;
        });
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        btn.disabled = false;
        btnText.classList.remove("hidden");
        spinner.classList.add("hidden");
    }
}

// Metrics Dashboard
async function loadMetrics() {
    document.getElementById("metricsLoading").classList.remove("hidden");
    document.getElementById("metricsDashboardContent").classList.add("hidden");
    
    try {
        const res = await fetch('/api/metrics');
        if(!res.ok) throw new Error("Failed to fetch metrics.");
        allMetrics = await res.json();
        renderMetricsDashboard();
    } catch(err) {
        showToast("Could not load metrics data.", "error");
    } finally {
        document.getElementById("metricsLoading").classList.add("hidden");
        document.getElementById("metricsDashboardContent").classList.remove("hidden");
    }
}

function renderMetricsDashboard() {
    const lang = document.getElementById("metricsLangSelect").value;
    const data = allMetrics[lang];
    
    if(!data || !data["Baseline"] || !data["Proposed"]) {
        document.getElementById("metricsTableBody").innerHTML = "<tr><td colspan='4' class='empty-state'>Data not available.</td></tr>";
        return;
    }
    
    const base = data["Baseline"];
    const prop = data["Proposed"];
    
    // KPI Cards
    document.getElementById("kpi-vocab").innerText = prop["Total Tokens"].toLocaleString();
    
    document.getElementById("kpi-cpt").innerText = prop["Chars per Token"].toFixed(2);
    const cptDiff = prop["Chars per Token"] - base["Chars per Token"];
    const cptEl = document.getElementById("kpi-cpt-trend");
    if(cptDiff > 0) {
        cptEl.innerHTML = `↑ ${(cptDiff).toFixed(2)} vs BPE`;
        cptEl.className = "kpi-trend trend-up";
    } else {
        cptEl.innerHTML = `↓ ${Math.abs(cptDiff).toFixed(2)} vs BPE`;
        cptEl.className = "kpi-trend trend-down";
    }
    
    document.getElementById("kpi-unk").innerText = (prop["Unknown Token Rate"] * 100).toFixed(2) + "%";
    document.getElementById("kpi-speed").innerText = prop["Encoding Time (s)"].toFixed(3) + "s";
    
    // Raw Data Table — supports Baseline, Byte-Level BPE (if present), and Proposed
    const tbody = document.getElementById("metricsTableBody");
    let tableRows = `
        <tr>
            <td>Baseline (BPE)</td>
            <td>${base["Total Tokens"].toLocaleString()}</td>
            <td>${base["Chars per Token"].toFixed(2)}</td>
            <td>${base["Uniform Encoding Cost (bits/char)"] !== undefined ? base["Uniform Encoding Cost (bits/char)"].toFixed(2) : '-'}</td>
            <td>${base["Encoding Time (s)"].toFixed(3)}s</td>
        </tr>`;
    
    const byteBpe = data["Byte-Level BPE"];
    if(byteBpe) {
        tableRows += `
        <tr>
            <td>Byte-Level BPE</td>
            <td>${byteBpe["Total Tokens"].toLocaleString()}</td>
            <td>${byteBpe["Chars per Token"].toFixed(2)}</td>
            <td>${byteBpe["Uniform Encoding Cost (bits/char)"] !== undefined ? byteBpe["Uniform Encoding Cost (bits/char)"].toFixed(2) : '-'}</td>
            <td>${byteBpe["Encoding Time (s)"].toFixed(3)}s</td>
        </tr>`;
    }
    
    tableRows += `
        <tr class="table-highlight">
            <td><strong>Proposed (CJK)</strong></td>
            <td><strong>${prop["Total Tokens"].toLocaleString()}</strong></td>
            <td><strong>${prop["Chars per Token"].toFixed(2)}</strong></td>
            <td><strong>${prop["Uniform Encoding Cost (bits/char)"] !== undefined ? prop["Uniform Encoding Cost (bits/char)"].toFixed(2) : '-'}</strong></td>
            <td><strong>${prop["Encoding Time (s)"].toFixed(3)}s</strong></td>
        </tr>`;
    
    tbody.innerHTML = tableRows;
    
    // Radar Chart
    renderRadarChart(base, prop);
}

function renderRadarChart(base, prop) {
    if(radarChart) radarChart.destroy();
    
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    // Normalize data for radar (higher is better for visualization)
    // CPT: higher is better (more compression)
    // Time: lower is better (so we invert: max_time - time)
    // UNK: lower is better (invert: 1 - UNK)
    
    const maxTime = Math.max(base["Encoding Time (s)"], prop["Encoding Time (s)"]) * 1.5;
    const bSpeed = maxTime - base["Encoding Time (s)"];
    const pSpeed = maxTime - prop["Encoding Time (s)"];
    
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Compression (Chars/Tok)', 'Speed (Inverted Time)', 'Vocabulary Size (Log scale)', 'Known Rate (1-UNK)'],
            datasets: [{
                label: 'Baseline (BPE)',
                data: [
                    base["Chars per Token"],
                    bSpeed,
                    Math.log10(base["Total Tokens"] || 1),
                    1 - base["Unknown Token Rate"]
                ],
                backgroundColor: 'rgba(113, 113, 122, 0.2)',
                borderColor: '#71717a',
                pointBackgroundColor: '#71717a'
            }, {
                label: 'Proposed (CJK)',
                data: [
                    prop["Chars per Token"],
                    pSpeed,
                    Math.log10(prop["Total Tokens"] || 1),
                    1 - prop["Unknown Token Rate"]
                ],
                backgroundColor: 'rgba(6, 182, 212, 0.2)',
                borderColor: '#06b6d4',
                pointBackgroundColor: '#06b6d4'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#a1a1aa', font: { family: 'Inter', size: 11 } },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: '#ededed', font: { family: 'Inter' } } }
            }
        }
    });
}

// Eval Dashboard
async function loadEvalCharts() {
    const lang = document.getElementById("evalLangSelect").value;
    document.getElementById("evalLoading").classList.remove("hidden");
    document.getElementById("evalDashboardContent").classList.add("hidden");
    
    try {
        const [bRes, pRes] = await Promise.all([
            fetch(`/api/eval/${lang}/baseline`).catch(() => null),
            fetch(`/api/eval/${lang}/proposed`).catch(() => null)
        ]);
        
        const bData = bRes && bRes.ok ? await bRes.json() : {};
        const pData = pRes && pRes.ok ? await pRes.json() : {};
        
        if(baseChart) baseChart.destroy();
        if(propChart) propChart.destroy();
        
        baseChart = createLineChart('baseChart', bData, '#71717a', 'Baseline Loss');
        propChart = createLineChart('propChart', pData, '#06b6d4', 'Proposed Loss');
        
        const execRes = await fetch(`/api/execute_results`).then(r => r.ok ? r.json() : {});
        renderExecuteResults(lang, execRes);
        
    } catch(err) {
        showToast("Error fetching evaluation charts.", "error");
    } finally {
        document.getElementById("evalLoading").classList.add("hidden");
        document.getElementById("evalDashboardContent").classList.remove("hidden");
    }
}

function createLineChart(id, data, color, label) {
    const ctx = document.getElementById(id).getContext('2d');
    const hasData = data.train_loss && data.train_loss.length > 0;
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: hasData ? Array.from({length: data.train_loss.length}, (_, i) => i + 1) : [1],
            datasets: [{
                label: hasData ? label : 'No Data',
                data: hasData ? data.train_loss : [0],
                borderColor: color,
                tension: 0.4,
                fill: true,
                backgroundColor: (context) => {
                    const ctx = context.chart.ctx;
                    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
                    gradient.addColorStop(0, `${color}40`);
                    gradient.addColorStop(1, `${color}00`);
                    return gradient;
                },
                borderWidth: 2,
                pointRadius: hasData && data.train_loss.length === 1 ? 4 : 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(10, 10, 10, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: { 
                    display: false // Cleaner look
                },
                y: { 
                    ticks: { color: '#71717a', font: { family: 'JetBrains Mono' } }, 
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    beginAtZero: true
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

function renderExecuteResults(lang, execRes) {
    const tbody = document.getElementById("executeTableBody");
    const res = execRes[lang];
    
    if(!res) {
        tbody.innerHTML = "<tr><td colspan='4' class='empty-state'>No EXECUTE data found for this language.</td></tr>";
        return;
    }
    
    const tasks = Object.keys(res.proposed || {}).filter(k => 
        (res.proposed[k].perplexity !== null && isFinite(res.proposed[k].perplexity)) || 
        (res.proposed[k].accuracy !== undefined && res.proposed[k].accuracy > 0)
    );
    
    if(tasks.length === 0) {
        tbody.innerHTML = "<tr><td colspan='4' class='empty-state'>All EXECUTE tasks failed or returned invalid data.</td></tr>";
        return;
    }
    
    const rows = tasks.map(task => {
        const prop = res.proposed[task];
        const base = res.baseline && res.baseline[task] ? res.baseline[task] : { error: "Failed" };
        
        const isAccuracy = prop.accuracy !== undefined;
        let pVal = isAccuracy ? (prop.accuracy * 100).toFixed(1) + "%" : prop.perplexity.toFixed(2);
        let bVal = base.perplexity ? base.perplexity.toFixed(2) : (base.accuracy !== undefined ? (base.accuracy * 100).toFixed(1) + "%" : "Failed");
        
        // CI string for accuracy tasks
        let ciStr = '';
        if(isAccuracy && prop.ci_95) {
            ciStr = ` <span class="text-sm" style="color:var(--text-tertiary)">[${(prop.ci_95[0]*100).toFixed(1)}–${(prop.ci_95[1]*100).toFixed(1)}%]</span>`;
        }
        
        // BPC for perplexity tasks
        let bpcStr = '-';
        if(!isAccuracy && prop.bpc !== null && prop.bpc !== undefined) {
            bpcStr = prop.bpc.toFixed(2);
        }
        
        // Determine winner
        let pWin = false;
        if(bVal !== "Failed") {
            if(isAccuracy) {
                pWin = prop.accuracy > base.accuracy;
            } else {
                pWin = prop.perplexity < base.perplexity;
            }
        } else {
            pWin = true;
        }
        
        return `<tr class="${pWin ? 'table-highlight' : ''}">
            <td><strong>${task.replace(/_/g, ' ').toUpperCase()}</strong></td>
            <td class="val-worst">${bVal}</td>
            <td class="val-best">${pWin ? '⭐ ' : ''}${pVal}${ciStr}</td>
            <td>${bpcStr}</td>
            <td><span class="badge">${isAccuracy ? 'ACCURACY' : 'PERPLEXITY'}</span></td>
        </tr>`;
    });
    
    tbody.innerHTML = rows.join("");
}

// Pipeline Status Check
async function loadStatus(lang_code) {
    try {
        const res = await fetch(`/api/status?lang_code=${lang_code}`);
        if(res.ok) {
            const data = await res.json();
            const updateStatus = (id, isReady) => {
                const dot = document.getElementById(id);
                if(isReady) {
                    dot.className = "status-dot ready";
                } else {
                    dot.className = "status-dot missing";
                }
            };
            updateStatus("dot-dataset", data.dataset);
            updateStatus("dot-tokenizer", data.tokenizer);
            updateStatus("dot-model", data.model);
            updateStatus("dot-metrics", data.metrics);
            updateStatus("dot-evals", data.evals);
        }
    } catch(e) {
        console.error("Status check failed", e);
    }
}

// Toasts
function showToast(message, type = "success") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    const icon = type === 'error' ? '⚠️' : '✅';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Ablation Study
async function loadAblation() {
    try {
        const res = await fetch('/api/ablation');
        if(!res.ok) return;
        const data = await res.json();
        
        if(!data || Object.keys(data).length === 0) return;
        
        // Use the first language with data
        const lang = document.getElementById("metricsLangSelect").value || Object.keys(data)[0];
        const langData = data[lang];
        if(!langData) return;
        
        document.getElementById("ablationSection").style.display = "block";
        renderAblationChart(langData);
    } catch(e) {
        console.error("Ablation fetch failed", e);
    }
}

function renderAblationChart(langData) {
    if(ablationChart) ablationChart.destroy();
    
    const ctx = document.getElementById('ablationChart').getContext('2d');
    const variants = Object.keys(langData).filter(k => !langData[k].error);
    const labels = variants.map(v => langData[v].description || v);
    
    const cptData = variants.map(v => langData[v]["Chars per Token"] || 0);
    const unkData = variants.map(v => (langData[v]["Unknown Token Rate"] || 0) * 100);
    const uecData = variants.map(v => langData[v]["Uniform Encoding Cost (bits/char)"] || 0);
    
    ablationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Chars/Token',
                    data: cptData,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: '#3b82f6',
                    borderWidth: 1
                },
                {
                    label: 'UNK Rate (%)',
                    data: unkData,
                    backgroundColor: 'rgba(239, 68, 68, 0.6)',
                    borderColor: '#ef4444',
                    borderWidth: 1
                },
                {
                    label: 'UEC (bits/char)',
                    data: uecData,
                    backgroundColor: 'rgba(6, 182, 212, 0.6)',
                    borderColor: '#06b6d4',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#ededed', font: { family: 'Inter' } } }
            },
            scales: {
                x: {
                    ticks: { color: '#a1a1aa', font: { family: 'Inter', size: 10 }, maxRotation: 45 },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#71717a' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    beginAtZero: true
                }
            }
        }
    });
}
