document.addEventListener("DOMContentLoaded", async () => {
  const session = requireSession();
  if (!session) return;

  const body = document.getElementById("history-body");

  try {
    const ads = await apiRequest(`/advertisements/user/${session.user_id}`);
    if (ads.length === 0) {
      body.innerHTML = '<tr><td colspan="4" class="muted">No submissions yet.</td></tr>';
      return;
    }
    body.innerHTML = ads.map(ad => `
      <tr>
        <td>${escapeHtml(ad.title)}</td>
        <td>${new Date(ad.created_at).toLocaleString()}</td>
        <td class="muted">Open to view</td>
        <td><a href="/analyse" class="btn secondary" style="padding:0.35rem 0.8rem;font-size:0.85rem;">Re-analyse</a></td>
      </tr>
    `).join("");
  } catch (err) {
    body.innerHTML = `<tr><td colspan="4" class="muted">Could not load history (${escapeHtml(err.message)}).</td></tr>`;
  }
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}
