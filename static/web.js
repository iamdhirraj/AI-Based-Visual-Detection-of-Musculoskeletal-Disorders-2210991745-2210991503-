const form = document.getElementById("predictForm");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const predictionLabel = document.getElementById("predictionLabel");
const predictionConfidence = document.getElementById("predictionConfidence");
const predictionBodyPart = document.getElementById("predictionBodyPart");
const predictionDisorder = document.getElementById("predictionDisorder");
const recommendationList = document.getElementById("recommendationList");
const template = document.getElementById("recommendationTemplate");

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  recommendationList.innerHTML = "";
  predictionLabel.textContent = "Analyzing image...";
  predictionConfidence.textContent = "";
  predictionBodyPart.textContent = "";
  predictionDisorder.textContent = "";
  results.classList.remove("hidden");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    const { prediction, recommendations } = payload;
    predictionLabel.textContent = prediction.has_disorder ? "Disorder detected" : "No disorder detected";
    predictionConfidence.textContent = prediction.body_part_confidence !== null
      ? `Body part confidence: ${(prediction.body_part_confidence * 100).toFixed(2)}%`
      : "Body part confidence: not available";
    predictionBodyPart.textContent = `Predicted body part: ${prediction.body_part}`;
    predictionDisorder.textContent = prediction.disorder_confidence !== null
      ? `Predicted disorder: ${prediction.disorder} (${(prediction.disorder_confidence * 100).toFixed(2)}%)`
      : `Predicted disorder: ${prediction.disorder}`;

    if (!prediction.has_disorder) {
      recommendationList.innerHTML = "<div class='exercise-card'><h3>No disorder detected</h3><p class='overview'>The image looks normal, so exercise recommendations are not shown.</p></div>";
      return;
    }

    recommendations.forEach((item) => {
      const clone = template.content.cloneNode(true);
      const card = clone.querySelector(".exercise-card");
      card.querySelector("h3").textContent = item.name || "Exercise";
      card.querySelector(".overview").textContent = item.overview || "No overview provided.";

      const meta = card.querySelector(".meta");
      [item.category, item.level, item.equipment, item.force].filter(Boolean).forEach((value) => {
        const pill = document.createElement("span");
        pill.textContent = value;
        meta.appendChild(pill);
      });

      const instructionList = card.querySelector(".instructions");
      (item.instructions || []).slice(0, 5).forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        instructionList.appendChild(li);
      });

      card.querySelector(".safety").textContent = item.safetyInfo || "Follow a pain-free, supervised progression.";
      recommendationList.appendChild(card);
    });
  } catch (error) {
    predictionLabel.textContent = "Prediction failed";
    predictionConfidence.textContent = error.message;
    recommendationList.innerHTML = "";
  }
});