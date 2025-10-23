// frontend/assets/script.js

// Tab switching
document.querySelectorAll('.tab-btn').forEach(button => {
  button.addEventListener('click', () => {
    const tabId = button.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(tabId).classList.add('active');
  });
});

// Form submission (index.html)
document.getElementById('uploadForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const progress = document.getElementById('progress');
  const progressBar = document.querySelector('.progress-bar');
  const submitBtn = e.target.querySelector('button[type="submit"]');

  submitBtn.disabled = true;
  submitBtn.textContent = 'Processing...';
  progress.style.display = 'block';

  try {
    const res = await fetch('http://127.0.0.1:8000/run', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Unknown error');
    }

    const data = await res.json();
    localStorage.setItem('lastResult', JSON.stringify(data));
    window.location.href = 'results.html';
  } catch (err) {
    alert('❌ Error: ' + err.message);
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run Feature Selection';
  }
});

// Load results (results.html)
document.addEventListener('DOMContentLoaded', () => {
  const resultContainer = document.getElementById('resultContainer');
  if (!resultContainer) return;

  const data = JSON.parse(localStorage.getItem('lastResult'));
  if (!data) {
    resultContainer.innerHTML = '<p>No results found. Please run a new analysis.</p>';
    return;
  }

  // Summary Table
  let tableHTML = `<table><thead><tr><th>Method</th><th># Features</th><th>MSE</th><th>Time</th></tr></thead><tbody>`;
  for (const [method, res] of Object.entries(data.results)) {
    const mse = res.mse !== null ? res.mse.toFixed(4) : 'N/A';
    tableHTML += `<tr><td>${method}</td><td>${res.selected.length}</td><td>${mse}</td><td>${res.time.toFixed(3)}s</td></tr>`;
  }
  tableHTML += `</tbody></table>`;

  // Features
  let featuresHTML = '';
  for (const [method, res] of Object.entries(data.results)) {
    featuresHTML += `<div class="feature-list"><strong>${method}:</strong> ${res.selected.join(', ') || 'None'}</div>`;
  }

  // Plots (with lightbox)
  let plotsHTML = '';
  data.plots.forEach(url => {
    plotsHTML += `
      <div style="text-align:center; margin:15px;">
        <img src="${url}" style="max-width:100%; height:auto; cursor:pointer;" onclick="openLightbox('${url}')">
      </div>
    `;
  });

  document.getElementById('summary').innerHTML = tableHTML;
  document.getElementById('features').innerHTML = featuresHTML;
  document.getElementById('plots').innerHTML = plotsHTML;
});

function openLightbox(src) {
  const lightbox = document.createElement('div');
  lightbox.className = 'lightbox';
  lightbox.innerHTML = `<img src="${src}">`;
  lightbox.onclick = () => document.body.removeChild(lightbox);
  document.body.appendChild(lightbox);
}