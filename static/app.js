// Navigation functionality
function initializeNavigation() {
  const navButtons = document.querySelectorAll('.nav-btn');
  const sections = document.querySelectorAll('section');

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      navButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sections.forEach(s => s.classList.add('hidden'));
      const sectionId = btn.id.replace('nav-', '') + '-section';
      document.getElementById(sectionId).classList.remove('hidden');
    });
  });
}

// Summary loading functionality
async function loadSummary() {
  try {
    const response = await fetch('/api/summary');
    const data = await response.json();
    renderSummary(data);
  } catch (error) {
    document.getElementById('summary-content').innerHTML = '<p class="error">Failed to load summary.</p>';
  }
}

function renderSummary(data) {
  document.getElementById('summary-content').innerHTML = `
    <p><strong>Total Patients:</strong> ${data.total_patients}</p>
    <h3>Averages</h3>
    <ul>${renderList(data.averages)}</ul>
    <h3>Minimums</h3>
    <ul>${renderList(data.mins)}</ul>
    <h3>Maximums</h3>
    <ul>${renderList(data.maxs)}</ul>
    <h3>Risk Distribution</h3>
    <ul>
      <li>Number of Endangered patients in Scenario 1: ${data.risk_distribution[0]}</li>
      <li>Number of Endangered patients in Scenario 2: ${data.risk_distribution[1]}</li>
    </ul>
  `;
}

function renderList(obj) {
  return Object.entries(obj)
    .map(([k, v]) => `<li>${k}: ${Number(v).toFixed(2)}</li>`)
    .join('');
}

// Visualizations loading functionality
async function loadVisualizations() {
  try {
    const response = await fetch('/api/visualizations');
    const data = await response.json();
    renderVisualizations(data);
  } catch (error) {
    console.error(error);
    document.getElementById('visualizations-content').innerHTML = '<p class="error">Failed to load visualizations.</p>';
  }
}

function renderVisualizations(data) {
  const container = document.getElementById('visualizations-content');
  container.innerHTML = '';

  // BMI vs Target scatter plot
  const bmiDiv = document.createElement('div');
  container.appendChild(bmiDiv);
  Plotly.newPlot(bmiDiv, data.bmi_target_scatter.data, data.bmi_target_scatter.layout);

  // BP vs Target scatter plot
  const bpDiv = document.createElement('div');
  container.appendChild(bpDiv);
  Plotly.newPlot(bpDiv, data.bp_target_scatter.data, data.bp_target_scatter.layout);

  // Pie charts
  const pieDiv1 = document.createElement('div');
  const pieDiv2 = document.createElement('div');

  container.appendChild(pieDiv1);
  const pieData1 = [{
    labels: data.risk_pie.x_array,
    values: data.risk_pie.y_array_s1,
    type: "pie",
    marker: { colors: ["#ee4444", "#4444ee"] }
  }];
  Plotly.newPlot(pieDiv1, pieData1, { title: "Risk Distribution Scenario 1" });

  container.appendChild(pieDiv2);
  const pieData2 = [{
    labels: data.risk_pie.x_array,
    values: data.risk_pie.y_array_s2,
    type: "pie",
    marker: { colors: ["#ee4444", "#4444ee"] }
  }];
  Plotly.newPlot(pieDiv2, pieData2, { title: "Risk Distribution Scenario 2" });
}

// Prediction form functionality
function initializePredictionForm() {
  const form = document.getElementById('prediction-form');
  const result = document.getElementById('result');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = Number(value);
    });

    result.classList.remove('hidden');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      renderPredictionResult(data);
    } catch (error) {
      result.innerHTML = `<p class="error">${error.message}</p>`;
    }
  });
}

function renderPredictionResult(data) {
  const result = document.getElementById('result');
  result.innerHTML = `
    <div class="result-grid">
      <div class="scenario-card">
        <h3>Scenario 1</h3>
        <p><strong>Risk:</strong> ${data.risk_label_s1}</p>
        <p><strong>Probability:</strong> ${Math.round(data.risk_probability_s1 * 100)}%</p>
      </div>
      <div class="scenario-card">
        <h3>Scenario 2</h3>
        <p><strong>Risk:</strong> ${data.risk_label_s2}</p>
        <p><strong>Probability:</strong> ${Math.round(data.risk_probability_s2 * 100)}%</p>
      </div>
    </div>
  `;
}

// Initialize the application
function initializeApp() {
  initializeNavigation();
  initializePredictionForm();
  loadSummary();
  loadVisualizations();
}

// Start the application when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);