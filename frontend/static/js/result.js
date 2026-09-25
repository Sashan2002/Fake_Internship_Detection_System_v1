document.addEventListener("DOMContentLoaded", async () => {
  const session = requireSession();
  if (!session) return;

  const loading = document.getElementById("loading");
  const content = document.getElementById("result-content");

  try {
    const prediction = await apiRequest(`/predictions/${window.PREDICTION_ID}`);
    renderVerdict(prediction);
    renderMetrics(prediction);

    const ad = await apiRequest(`/advertisements/${prediction.advertisement_id}`);
    document.title = `Result: ${ad.title}`;

    const credibility = await apiRequest("/analyse-credibility", {
      method: "POST",
      body: JSON.stringify({ ...ad, advertisement_id: ad.advertisement_id }),
    });
    renderCredibility(credibility);

    const explanation = await apiRequest(`/explanations/${prediction.prediction_id}`);
    renderExplanation(explanation);

    loading.classList.add("hidden");
    content.classList.remove("hidden");
  } catch (err) {
    loading.textContent = `Could not load result: ${err.message}`;
  }
});

function renderVerdict(prediction) {
  const verdict = document.getElementById("verdict");
  const label = document.getElementById("verdict-label");
  const sub = document.getElementById("verdict-sub");
  const meta = document.getElementById("prediction-meta");

  const cls = prediction.predicted_class === "Potentially Legitimate" ? "legitimate"
    : prediction.predicted_class === "Potentially Fraudulent" ? "fraudulent" : "review";
  verdict.classList.add(cls);
  label.textContent = prediction.predicted_class;
  sub.textContent = cls === "review"
    ? "The model's confidence was too low for a clear prediction. Human review is recommended."
    : "This is the model's prediction, not confirmed fact.";
  meta.textContent = `Model ${prediction.model_version} · ${new Date(prediction.created_at).toLocaleString()}`;
}

function renderMetrics(prediction) {
  setBar("probability", prediction.fraud_probability);
  setBar("confidence", prediction.confidence);
  setBar("uncertainty", prediction.uncertainty ?? 0);
}

function setBar(name, value) {
  const pct = Math.round((value ?? 0) * 100);
  document.getElementById(`bar-${name}`).style.width = `${pct}%`;
  document.getElementById(`value-${name}`).textContent = `${pct}%`;
}

function renderCredibility(result) {
  const grid = document.getElementById("credibility-grid");
  const note = document.getElementById("credibility-note");
  const vector = result.feature_vector || {};

  const presenceEntries = Object.entries(vector).filter(([k]) => k.endsWith("_present"));
  grid.innerHTML = presenceEntries.map(([key, val]) => `
    <div class="indicator">
      <span>${formatLabel(key)}</span>
      <span class="value ${val ? "present" : "absent"}">${val ? "Present" : "Absent"}</span>
    </div>
  `).join("");

  note.textContent = result.disclaimer || "";
}

function renderExplanation(explanation) {
  const list = document.getElementById("evidence-list");
  const note = document.getElementById("explanation-note");

  if (!explanation.available) {
    list.innerHTML = `<li class="evidence-item"><span class="muted">${explanation.reason || "Not available."}</span></li>`;
  } else {
    const factors = explanation.top_factors || [];
    list.innerHTML = factors.map(f => `
      <li class="evidence-item">
        <span class="token">${escapeHtml(f.token || f.feature)}</span>
        <span class="direction ${f.direction}">${f.direction === "increases_fraud_signal" ? "↑ fraud signal" : "↓ fraud signal"}</span>
      </li>
    `).join("") || '<li class="evidence-item"><span class="muted">No strongly influential factors identified.</span></li>';
  }
  note.textContent = explanation.disclaimer || "";
}

function formatLabel(key) {
  return key.replace(/_/g, " ").replace("present", "").trim().replace(/^\w/, c => c.toUpperCase());
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}
