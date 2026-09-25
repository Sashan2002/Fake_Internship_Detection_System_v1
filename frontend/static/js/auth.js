document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("auth-form");
  const toggle = document.getElementById("toggle-mode");
  const nameField = document.getElementById("name-field");
  const title = document.getElementById("form-title");
  const submitBtn = document.getElementById("submit-btn");
  const errors = document.getElementById("errors");

  let mode = "login"; // or "register"

  toggle.addEventListener("click", (e) => {
    e.preventDefault();
    mode = mode === "login" ? "register" : "login";
    const isRegister = mode === "register";
    nameField.classList.toggle("hidden", !isRegister);
    document.getElementById("name").required = isRegister;
    title.textContent = isRegister ? "Register" : "Sign in";
    submitBtn.textContent = isRegister ? "Create account" : "Sign in";
    toggle.textContent = isRegister ? "Already have an account? Sign in" : "Need an account? Register";
    showErrors(errors, null);
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    showErrors(errors, null);
    submitBtn.disabled = true;

    const payload = {
      email: document.getElementById("email").value.trim(),
      password: document.getElementById("password").value,
    };
    if (mode === "register") {
      payload.name = document.getElementById("name").value.trim();
    }

    try {
      if (mode === "register") {
        await apiRequest("/auth/register", { method: "POST", body: JSON.stringify(payload) });
        // After registering, log in immediately for a smoother flow.
        const session = await apiRequest("/auth/login", { method: "POST", body: JSON.stringify(payload) });
        setSession(session);
      } else {
        const session = await apiRequest("/auth/login", { method: "POST", body: JSON.stringify(payload) });
        setSession(session);
      }
      window.location.href = "/dashboard";
    } catch (err) {
      showErrors(errors, [err.message]);
    } finally {
      submitBtn.disabled = false;
    }
  });
});
