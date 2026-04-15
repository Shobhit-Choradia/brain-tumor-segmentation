const BACKEND_URL = "https://brain-tumor-segmentation-project.onrender.com";
let reportData = null;
let selectedFiles = {};

// ✅ ONLY 4 FILES NOW
const REQUIRED = ['t1','t1ce','t2','flair'];

function filesSelected() {
    const input = document.getElementById('files');
    const files = Array.from(input.files);
    const list = document.getElementById('file-list');
    const box = document.querySelector('.upload-box');
    const text = document.getElementById('upload-text');

    selectedFiles = {};
    list.innerHTML = '';

    files.forEach(f => {
        const name = f.name.toLowerCase();
        let key = null;
        if (name.includes('_t1ce')) key = 't1ce';
        else if (name.includes('_t1')) key = 't1';
        else if (name.includes('_t2')) key = 't2';
        else if (name.includes('_flair')) key = 'flair';

        if (key) {
            selectedFiles[key] = f;
            const tag = document.createElement('div');
            tag.className = 'file-tag ok';
            tag.textContent = `✅ ${key.toUpperCase()}: ${f.name}`;
            list.appendChild(tag);
        }
    });

    const count = Object.keys(selectedFiles).length;
    text.innerHTML = `
        ${count}/4 files selected
        <span>${count === 4 ? '✅ Ready to analyze!' : '⚠️ Need all 4 files'}</span>
    `;
    if (count === 4) box.classList.add('ready');
}

async function analyze() {
    // ✅ CHECK ONLY 4 FILES
    const missing = REQUIRED.filter(k => !selectedFiles[k]);
    if (missing.length > 0) {
        alert(`Missing files: ${missing.join(', ').toUpperCase()}\nPlease select all 4 .nii files!`);
        return;
    }

    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const formData = new FormData();
    for (const [key, file] of Object.entries(selectedFiles)) {
        formData.append('file', file);
    }

    try {
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method : 'POST',
            body : formData
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        displayResults(data);
    } catch(err) {
        alert(`Error: ${err.message}`);
        console.error(err);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    document.getElementById('seg-image').src = `data:image/png;base64,${data.segmentation_image}`;

    // Severity
    const badge = document.getElementById('severity-badge');
    const labels = {
        HIGH:'⚠️ HIGH RISK',
        MODERATE:'⚡ MODERATE RISK',
        LOW:'✅ LOW RISK'
    };
    badge.textContent = labels[data.severity];
    badge.className = `severity-badge severity-${data.severity}`;

    // Volumes
    const colors = {necrotic:'#f87171', edema:'#34d399', enhancing:'#fbbf24'};
    const maxVol = Math.max(...[
        data.volumes.necrotic,
        data.volumes.edema,
        data.volumes.enhancing
    ]) || 1;

    const volDiv = document.getElementById('volume-bars');
    volDiv.innerHTML = '';
    for (const [key, val] of Object.entries(data.volumes)) {
        if (key === 'total') continue;
        const pct = ((val / maxVol) * 100).toFixed(1);
        const name = key.charAt(0).toUpperCase() + key.slice(1);
        volDiv.innerHTML += `
            <div class="volume-bar-item">
                <div class="volume-bar-label">
                    <span>${name}</span><span>${val} cm³</span>
                </div>
                <div class="volume-bar-track">
                    <div class="volume-bar-fill" style="width:${pct}%;background:${colors[key]}"></div>
                </div>
            </div>`;
    }
    volDiv.innerHTML += `<div class="volume-total">Total Tumor: ${data.volumes.total} cm³</div>`;

    // Report
    const now = new Date().toLocaleString();
    reportData = `
BRAIN TUMOR SEGMENTATION REPORT
Date : ${now}
Analyzed Slice : ${data.slice_idx}
----------------------------------------
Tumor Volumes
----------------------------------------
Necrotic Core : ${data.volumes.necrotic} cm³
Edema : ${data.volumes.edema} cm³
Enhancing Tumor : ${data.volumes.enhancing} cm³
----------------------------------------
TOTAL : ${data.volumes.total} cm³
Severity : ${data.severity}
Note: AI-generated result (not for clinical use)
    `;
    document.getElementById('report').textContent = reportData;
    document.getElementById('results').scrollIntoView({behavior:'smooth'});
}

function downloadReport() {
    if (!reportData) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([reportData], {type:'text/plain'}));
    a.download = 'tumor_report.txt';
    a.click();
}

