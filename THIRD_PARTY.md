# Source attribution

The method and network architecture are from Zikun Ye, Zhiqi Zhang, Dennis J. Zhang, Heng Zhang, and Renyu Zhang, *Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence*, Management Science. [Paper and DOI](https://doi.org/10.1287/mnsc.2024.04625).

`build_network()` in `dedl/tf_model.py` adapts `sig_dnn()` from the authors' `main.ipynb`, zero-based cell 12. It retains the layer operations, order and named outputs while incorporating the builder directly into the training module and omitting the printed model summary.

- [Upstream source](https://github.com/zikunye2/deep_learning_based_causal_inference_for_combinatorial_experiments/tree/054ec6ae3541474a189203a89fc34698432735f7)
- Source commit: `054ec6ae3541474a189203a89fc34698432735f7`

No project-wide license is applied to third-party material; consult the upstream source and authors for reuse terms. The synthetic experiment, PyTorch implementation, score calculation, entry scripts and documentation are this repository's implementation, not the authors' released empirical pipeline.
