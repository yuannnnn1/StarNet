# Pilot gate passed

2026-09-19. Device: Apple M5 MPS, with CPU checkpoint/forward verification.
Two epochs per model, 2048 training images, 512 validation and 512 test images.
Final pilot training durations were about 2.7 s baseline, 3.1 s StarNet and
2.9 s additive ablation. Dataset download took about 5 minutes. Data preparation,
plots and independent verification are additional and excluded from training time.

All training losses decreased. Final pilot validation accuracies were 29.10%,
33.40% and 34.18%, respectively. These are pipeline checks, not final results.
Actual values and checkpoint evidence live in artifacts/pilot/results.
Eleven figures were generated and the loss figure visually inspected. Dashboard
JSON was generated and equality-checked against saved results. Browser testing
is deferred until final experiment data are available.

Corrections before full training: compute normalization with float64 accumulation;
increase baseline widths from 24/48/96 to 32/64/128 for 289194 parameters versus
280522 in StarNet and ablation. Corrected pilots rerun successfully. The initial
pilot files were superseded by final corrected pilot outputs. No full run used
the initial settings. CPU sandbox cannot see MPS; authorized runs outside the
sandbox use the GPU. A guessed upstream source path returned 404 and was fixed
by inspecting the repository tree. Plotting initially needed a writable font cache.

Gate evidence: artifacts/pilot/results/verification.json. Based on warm epoch
timings scaled to 45000 training examples, estimated total full training for
three models and 20 epochs is around 25 minutes. Keep the planned 20 epochs,
seed 3024, batch 128, AdamW lr 0.003, weight decay 0.0005, cosine schedule.
No pilot test scores were used to tune architecture for accuracy: the baseline
width change was solely to match parameter count. The first 512 official test
images were accessed for pilot pipeline evaluation; the 10000-image test set
remains excluded from fitting, normalization and checkpoint/budget selection.
