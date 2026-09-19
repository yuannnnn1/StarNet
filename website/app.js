const names = {
  baseline: "Conventional CNN",
  starnet: "StarNet",
  ablation: "Additive ablation",
};
const pct = (x) => (x * 100).toFixed(2) + "%";
let data;
function confusion() {
  const name = document.querySelector("#confusion-model").value;
  const run = data.runs.find((r) => r.name === name);
  const img = document.querySelector("#confusion-chart");
  img.src = `../artifacts/figures/confusion_matrix_${name}.png`;
  img.alt =
    names[name] +
    " test confusion matrix; rows are true labels and columns predicted labels";
  document.querySelector("#class-recall").innerHTML =
    "<table><thead><tr><th>Class</th>" +
    data.dataset.classes.map((c) => `<th>${c}</th>`).join("") +
    "</tr></thead><tbody><tr><td>Recall</td>" +
    run.class_recall.map((x) => `<td>${pct(x)}</td>`).join("") +
    "</tr></tbody></table>";
}
function predictions() {
  const name = document.querySelector("#prediction-model").value,
    filter = document.querySelector("#prediction-filter").value;
  document.querySelector("#examples").innerHTML = data.examples[name]
    .filter((e) => filter === "all" || e.correct === (filter === "correct"))
    .map(
      (e) =>
        `<article class="example"><img src="../artifacts/figures/${e.image}" alt="CIFAR-10 test image ${e.index}, true label ${e.true}" loading="lazy"><div><strong class="${e.correct ? "correct" : "incorrect"}">${e.correct ? "Correct" : "Incorrect"} · #${e.index}</strong>True: ${e.true}<br>Predicted: ${e.predicted}<br>Confidence: ${pct(e.confidence)}</div></article>`,
    )
    .join("");
}
async function init() {
  try {
    const response = await fetch("../artifacts/results/dashboard.json");
    if (!response.ok) throw Error("Experiment results are not available yet.");
    data = await response.json();
    if (data.pilot)
      throw Error(
        "Pilot data cannot be displayed as final experiment results.",
      );
    const star = data.runs.find((r) => r.name === "starnet"),
      add = data.runs.find((r) => r.name === "ablation"),
      cfg = star.config;
    document.querySelector("#status").textContent = "";
    document.querySelector("#summary").innerHTML =
      `<div><strong>${pct(star.test_accuracy)}</strong><span>StarNet test accuracy</span></div><div><strong>${star.parameters.toLocaleString()}</strong><span>StarNet parameters</span></div><div><strong>${cfg.epochs}</strong><span>Epochs per model · seed ${cfg.seed}</span></div>`;
    document.querySelector("#device").textContent =
      `${data.environment.hardware} · ${data.environment.device.toUpperCase()} · PyTorch ${data.environment.torch}`;
    document.querySelector("#split").innerHTML = ["train", "validation", "test"]
      .map(
        (k) =>
          `<div><strong>${data.dataset[k + "_count"].toLocaleString()}</strong><span>${k} images</span></div>`,
      )
      .join("");
    document.querySelector("#preprocessing").textContent =
      `Training: random crop with 4-pixel padding and horizontal flip. Normalization is measured only on the training split: mean [${data.dataset.mean.map((x) => x.toFixed(4)).join(", ")}], standard deviation [${data.dataset.std.map((x) => x.toFixed(4)).join(", ")}]. Validation and test use normalization only.`;
    document.querySelector("#training-config").textContent =
      `AdamW · initial learning rate ${cfg.learning_rate} · weight decay ${cfg.weight_decay} · cosine schedule · batch size ${cfg.batch_size} · ${cfg.epochs} epochs · cross-entropy loss.`;
    document.querySelector("#comparison-body").innerHTML = data.runs
      .map(
        (r) =>
          `<tr><td>${names[r.name]}</td><td>${pct(r.test_accuracy)}</td><td>${r.test_loss.toFixed(4)}</td><td>${r.parameters.toLocaleString()}</td><td>${(r.conv_linear_macs / 1e6).toFixed(2)} M</td><td>${(r.training_seconds / 60).toFixed(2)} min</td><td>${r.inference_ms_per_image.toFixed(4)} ms</td></tr>`,
      )
      .join("");
    document.querySelector("#timing-note").textContent =
      `Inference: ${star.inference_protocol}; batch size ${cfg.batch_size}. Per-image figure is amortized batch time, not single-image latency. Training time includes validation and checkpoint writing. MACs count convolution and linear layers only; they exclude normalization, activation, pooling and branch combination.`;
    const delta = (star.test_accuracy - add.test_accuracy) * 100;
    document.querySelector("#ablation-result").textContent =
      `StarNet ${pct(star.test_accuracy)} vs. addition ${pct(add.test_accuracy)}: ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} percentage points with multiplication in this run.`;
    document.querySelector("#metric").addEventListener("change", (e) => {
      const img = document.querySelector("#training-chart");
      img.src = `../artifacts/figures/training_${e.target.value}.png`;
      img.alt = `Training and validation ${e.target.value} curves for all models`;
    });
    document
      .querySelector("#confusion-model")
      .addEventListener("change", confusion);
    document
      .querySelector("#prediction-model")
      .addEventListener("change", predictions);
    document
      .querySelector("#prediction-filter")
      .addEventListener("change", predictions);
    confusion();
    predictions();
    const repositoryResponse = await fetch("../docs/repository.json");
    if (repositoryResponse.ok) {
      const repository = await repositoryResponse.json();
      const status = document.querySelector("#repository-status");
      status.replaceChildren();
      const link = document.createElement("a");
      link.href = repository.url;
      link.textContent = "Assignment source-code repository";
      status.append(
        link,
        document.createTextNode(
          " · Local source publication deferred by the student.",
        ),
      );
    }
  } catch (error) {
    document.querySelector("#status").textContent = error.message;
  }
}
init();
