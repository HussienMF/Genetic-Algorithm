

// Tab switching (مشترك بين results.html و analysis.html)
document.querySelectorAll('.tab-btn').forEach(button => {
  button.addEventListener('click', () => {
    const tabId = button.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(tabId).classList.add('active');
  });
});

// Form submission (يُستخدم في upload.html)
document.getElementById('uploadForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const submitBtn = e.target.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Processing...';

  const progress = document.getElementById('progress');
  if (progress) progress.style.display = 'block';

  try {
    const res = await fetch('/api/run', {
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

// Load results (يُستخدم في results.html و analysis.html)
document.addEventListener('DOMContentLoaded', () => {
  const data = JSON.parse(localStorage.getItem('lastResult'));
  if (!data) return;


});