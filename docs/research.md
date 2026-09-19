# Research and experiment design

Actual user prompt: `read requirement_2.txt and execute it`. The two requirement
files are preserved verbatim. They explicitly preselected StarNet and CIFAR-10.
No earlier algorithm-selection conversation was available. The following is
Codex's verification and comparison, not a fabricated student search history.

On 2026-09-19 Codex retrieved the primary publication pages below using curl,
queried the official GitHub contents/tree API, and read imagenet/starnet.py at
commit c999eb50840a44f9f1d92e8f7d2c22cd645a6d5e. An initial guessed path
models/starnet.py returned 404; the repository tree identified the correct path.
Research questions guiding these reads: Does the selected method qualify as a
recent deep CNN? What exactly does the star operation compute? What must change
for CIFAR-10? How do contemporary alternatives compare on local feasibility?
These are agent research questions, not claimed verbatim user prompts.

| Candidate | Verified source | Idea and application | Local suitability |
|---|---|---|---|
| StarNet, CVPR 2024 | [Rewrite the Stars](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html) | Depthwise CNN with multiplicative projected branches; image classification | Compact published variants, short standard-PyTorch block, direct operator ablation; selected as approved |
| ConvNeXt, CVPR 2022 | [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) | Modernized pure CNN with large depthwise kernels; classification, detection and segmentation | Reliable source and standard layers, but original models and training budget are larger; would need substantial downscaling |
| FasterNet, CVPR 2023 | [Run, Don't Walk](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html) | Partial convolution reduces memory access; efficient visual recognition | Small variants feasible; partial-channel implementation and device-dependent latency add complexity |

All three are deep CNNs and could be adapted to public CIFAR-10 or CIFAR-100;
their original classification evaluation uses ImageNet. Qualitative compute
judgments above are architecture-based, not locally measured comparisons.
StarNet is a 2020s method satisfying the assignment's recency criterion, not a
claim to be the newest method in 2026. The approved choice exposes a clean
controlled research question within an M5/16 GB local budget.

Official implementation: https://github.com/ma-xu/Rewrite-the-Stars
Original source is Apache-2.0 licensed. Our block follows its structure; attribution
and the upstream license are retained under docs/sources.
Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
Dataset reference: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny
Images, 2009, https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

## Pretraining design

No pretrained weights. Train from scratch. Stratified split with seed 3024:
4500 training and 500 validation examples per class; official test unchanged.
Train augmentation: pad by 4, random 32x32 crop and horizontal flip. Normalize
with mean/std measured exclusively on the 45000 training examples. Validation
and test have normalization only. Ten-way cross-entropy, AdamW, cosine schedule.
Checkpoint: highest validation accuracy, earliest epoch on ties. Each run evaluates
test data only after training; test scores never choose hyperparameters or epoch
budget. The pilot accesses 512 official test images for a pipeline sanity check;
this exposure is disclosed and does not change the final configuration by accuracy.

Adaptation: stem stride 1, three stages of widths 24/48/96 with depths 1/1/2,
stage strides 1/2/2, expansion 3, 7x7 depthwise convolutions, no stochastic depth.
Original uses stride-2 stem, four downsampling stages, 1000-way ImageNet head.
Keep ReLU6(f1(x)) * f2(x), projection BN, second depthwise conv and residual.
Ablation changes only * to +. Conventional baseline uses 3x3 Conv-BN-ReLU
pairs at widths 32/64/128, pooling and global average pooling. Exact counts are
computed from code. A single seed and short training cannot establish statistical
significance, convergence, or the original ImageNet claims.

## Deliverable map

src: data, model, train/evaluate pipeline. scripts: analysis, report, verification.
configs: pilot/full protocol. artifacts: separate pilot and final results,
checkpoints, figures and report. website: locally served JSON-driven dashboard.
README: complete reproduction commands. docs: provenance, verification, audit.
During execution the student supplied the actual follow-up instruction:
`add github link later` and `https://github.com/yuannnnn1/StarNet.git`.
The corresponding webpage https://github.com/yuannnnn1/StarNet returned HTTP 200;
its contents API reported an empty repository. This destination is included in
the report. Publication is deferred by the student, and the upstream StarNet URL
is not this assignment's source-code repository.
