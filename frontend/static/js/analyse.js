document.addEventListener("DOMContentLoaded", () => {
  const session = requireSession();
  if (!session) return;

  const form = document.getElementById("analyse-form");
  const errors = document.getElementById("errors");
  const submitBtn = document.getElementById("submit-btn");
  const processing = document.getElementById("processing");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    showErrors(errors, null);

    const payload = {
      user_id: session.user_id,
      title: document.getElementById("title").value.trim(),
      description: document.getElementById("description").value.trim(),
      requirements: document.getElementById("requirements").value.trim(),
      benefits: document.getElementById("benefits").value.trim(),
      location: document.getElementById("location").value.trim(),
      salary_range: document.getElementById("salary_range").value.trim(),
      employer_name: document.getElementById("employer_name").value.trim(),
      company_profile: document.getElementById("company_profile").value.trim(),
      industry: document.getElementById("industry").value.trim(),
      employment_type: document.getElementById("employment_type").value.trim(),
      has_company_logo: document.getElementById("has_company_logo").checked,
      has_questions: document.getElementById("has_questions").checked,
    };

    submitBtn.disabled = true;
    processing.classList.remove("hidden");

    try {
      const { advertisement_id } = await apiRequest("/advertisements", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      const prediction = await apiRequest("/predict", {
        method: "POST",
        body: JSON.stringify({ ...payload, advertisement_id }),
      });

      if (prediction.prediction_id) {
        window.location.href = `/result/${prediction.prediction_id}`;
      } else {
        // Model available but advertisement wasn't persisted with a linked
        // prediction row (shouldn't normally happen given advertisement_id
        // was supplied) - fall back to showing the raw response.
        showErrors(errors, ["Prediction completed but could not be linked to a saved record."]);
      }
    } catch (err) {
      showErrors(errors, [err.message]);
    } finally {
      submitBtn.disabled = false;
      processing.classList.add("hidden");
    }
  });
});
