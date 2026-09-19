# Requirement audit

Authoritative specification: requirement.txt, with the approved topic, pilot gate,
website and execution details in requirement_2.txt. Both original files are preserved.
Final local verification completed on 2026-09-19. Explicit exceptions are listed below.

| Requirement area | Evidence and status |
|---|---|
| 1. Recent deep CNN with a vision application | StarNet, CVPR 2024; original paper and official implementation verified; CIFAR-10 classification |
| 2. AI research and proposal | docs/research.md records real prompt, candidate comparison, primary URLs, actual retrieval/debugging and approved design; no invented prior conversation |
| 3. Legitimate feasible dataset | Official CIFAR-10 download with torchvision integrity checks; seeded class-stratified 45000/5000 split and official 10000 test images |
| 4. AI implementation | src/ implements data, all three models, training, validation and testing; scripts/ and website/ implement analysis and visualization; student supplied requirements, not code |
| 5. Reproducibility | configs/, pinned requirements.txt, results/environment.json, source hashes, split indices, training-only normalization and checkpoint hashes |
| 6. Actual experiments | All three 20-epoch MPS runs completed; test accuracy CNN 87.22%, StarNet 86.14%, addition 85.60%; raw results and checkpoints retained |
| 7. Visualizations | Eleven final figures and eleven separate pilot figures generated from actual outputs; final dashboard/curve screenshots inspected |
| 8. Verification | Pilot, full protocol, checkpoint/metric consistency, report numerical audit, browser controls and 36 HTTP resources passed |
| 9. Six-section report | DOCX and Markdown generated with all six headings and real repository URL; Markdown replaces PDF at user request; DOCX page-render QA unavailable |
| 10. Evidence and traceability | Original requirements, research sources, pilot gate, raw histories, predictions, configuration and result-driven report generator |
| 11. Clean repository | Local Git main branch with supplied origin; ignores data, environments, checkpoints, caches and QA intermediates; remote push deferred by user |
| 12. Required workflow | Inspect and research preceded implementation; verified pilot preceded full training; report follows rereading authoritative requirements |
| 13. Deliverables | Source, configurations, results, figures, functional website, README, Word and Markdown reports present; source publication deferred |

## Specific protocol checks

artifacts/results/protocol_checks.json verifies exactly 4500/500/1000 images per
class, identical seeded StarNet/addition initial weights, and finite CPU
forward/backward gradients for all models. The pilot checks separately verify
disjoint/exhaustive indices, checkpoint integrity/loading, reconstructed accuracy
and confusion matrices, and equality of dashboard data to raw results.

The pilot evaluated the first 512 official test images solely for pipeline sanity
checking. This exposure is disclosed; test scores did not select the architecture
by accuracy, full epoch budget or checkpoints. The baseline width correction was
for parameter comparability. There is no claim of a completely unobserved test
set before all pipeline checks.

## Publication status

The student supplied https://github.com/yuannnnn1/StarNet.git and requested adding
the link with publication left for later. Its webpage returned HTTP 200; GitHub's
contents API reports an empty repository. The local origin points to it. The
report includes the actual webpage URL, not the upstream paper's repository as a
substitute. Uploading the local implementation remains deliberately deferred.

## Scientific limitations

One seed, one dataset, twenty epochs, no statistical significance test, no
hyperparameter search and no original ImageNet reproduction. MPS is not guaranteed
bitwise deterministic. The conventional CNN differs structurally despite a close
parameter budget. The additive ablation also changes activation scale. MACs cover
Conv/Linear layers only. Inference cost is amortized batch throughput, not
single-image deployment latency. Prediction confidence is uncalibrated.

## Report format and rendering limitation

The user instructed: `if it's hard to generate pdf file, try markdown file instead`.
Accordingly, artifacts/report/StarNet_Report.md is provided alongside the DOCX.
The document skill's render_docx.py was attempted and failed because LibreOffice
was absent. Native Pages access timed out; a subsequent export attempt failed
with a missing-document error. No PDF was generated or claimed. An official
LibreOffice installer was downloaded to temporary storage during troubleshooting
but was not installed or used; this approach was abandoned at the user's direction.
The DOCX passed XML/structure, embedded-image hash and numerical checks, but no
claim is made that its paginated layout was visually verified. Review in Word
before submitting the Word version. The Markdown report has valid relative image
links, matching metrics and exactly the six requested top-level content sections.

## Final evidence files

- artifacts/results/verification.json: saved metrics, confusion matrices and checkpoint checks.
- artifacts/results/protocol_checks.json: split class counts and seeded model controls.
- artifacts/results/website_ui_checks.json: desktop, 390px/320px mobile, controls and decoded images.
- artifacts/results/website_http_checks.json: 36 local URLs returned nonempty HTTP 200.
- artifacts/report/report_audit.json: six headings, real repository hyperlink, figure hashes and result values.
- artifacts/report/report_evidence.json: result sources and calculated comparison differences.
- artifacts/pilot/results/verification.json: mandatory pilot gate, rerun after standalone evaluation.

Python source compilation, JavaScript syntax and pip dependency checks passed.
The standalone --evaluate-only command was executed successfully on the saved
pilot baseline. Full-result files were not overwritten by that command. Training
source hashes still matched recorded execution at final review.

Remaining external task: publish the local source to the user-supplied repository
when the student is ready. No remote push was performed.
