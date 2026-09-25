/**
 * Small fetch wrapper + local session helper.
 * The "session" here is intentionally minimal (localStorage of user_id) -
 * this is a decision-support tool demo, not a production auth system.
 */
const API_BASE = "/api";

async function apiRequest(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = (data.errors && data.errors.join(", ")) || `Request failed (${res.status})`;
    throw new Error(message);
  }
  return data;
}

function getSession() {
  const raw = localStorage.getItem("session");
  return raw ? JSON.parse(raw) : null;
}

function setSession(session) {
  localStorage.setItem("session", JSON.stringify(session));
}

function requireSession() {
  const session = getSession();
  if (!session) {
    window.location.href = "/";
  }
  return session;
}

function showErrors(container, errors) {
  if (!errors || errors.length === 0) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }
  container.classList.remove("hidden");
  container.innerHTML = Array.isArray(errors) ? errors.join("<br>") : String(errors);
}
