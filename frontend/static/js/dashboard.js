document.addEventListener("DOMContentLoaded", async () => {
  const session = requireSession();
  if (!session) return;

  const list = document.getElementById("recent-list");
  try {
    const ads = await apiRequest(`/advertisements/user/${session.user_id}`);
    if (ads.length === 0) {
      list.innerHTML = '<p class="muted">No submissions yet. <a href="/analyse">Analyse your first advertisement</a>.</p>';
      return;
    }
    const rows = ads.slice(0, 5).map(ad => `
      <div class="indicator">
        <span>${escapeHtml(ad.title)}</span>
        <span class="muted">${new Date(ad.created_at).toLocaleDateString()}</span>
      </div>
    `).join("");
    list.innerHTML = rows;
  } catch (err) {
    list.innerHTML = `<p class="muted">Could not load recent submissions (${escapeHtml(err.message)}).</p>`;
  }
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}
