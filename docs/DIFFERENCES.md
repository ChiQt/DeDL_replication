# Configuration and source comparison / 配置与来源对照

This project reproduces the **comparison** in [Figure 7](https://rphilipzhang.github.io/rphilipzhang/DeDL-nonblind.pdf), not the numerical curve from Platform O's private data. It combines the main text's treatment design and network with a fixed synthetic response. The [Online Appendix](https://rphilipzhang.github.io/rphilipzhang/DeDL-nonblind_Appendix.pdf) and [public notebook](https://github.com/zikunye2/deep_learning_based_causal_inference_for_combinatorial_experiments) are distinct reference sources, not a complete public implementation of the empirical figure.

| Component | Here | Source and limit |
|---|---|---|
| Treatments | Three binary treatments; five observed groups | Main-text experiment. Hidden groups `110`, `101`, `011` have no outcomes in estimation data. |
| Covariates | 10 continuous and 16 categorical, encoded as 87 inputs | Raw counts follow the main text. Dummy-column names and widths follow the public notebook, not a literal coding of Appendix Table A1: for example, the appendix describes five age categories and three gender categories, while the notebook lists seven age columns and two gender columns. Our independent uniform category draws are a project choice; categories do not affect the response. |
| Response | Fixed coefficient matrix, sinusoidal/linear basis, scaled sigmoid, uniform noise $U(-0.5,0.5)$ | Project DGP in `response_parameters.json`; not the platform mechanism or the Appendix's cubic simulation. |
| Sample size | 20,000 users | Computational choice. The empirical sample is far larger. |
| Network | Two 20–20–20 ReLU branches and scaled sigmoid | Main text and the adapted public TensorFlow builder. |
| Training | 400 epochs, batch 100, Adam learning rate $10^{-4}$ | Main-text horizon and selected hyperparameters. Exact empirical minibatch order and optimizer state are unavailable. |
| Four folds | Each fold scores held-out users; 10% of the other folds is validation | Four-fold cross-fitting follows the paper. The 10% inner split is a project choice. |
| Figure | Four-fold pooled ATE at every checkpoint, then MAPE | The paper does not disclose whether Figure 7 pools all four folds or shows a selected fold. Our aggregation is explicit. |
| Ridge | Main 0.0005; diagnostic 0.05 | 0.0005 is an Appendix value, not a stated final Figure 7 ridge. The notebook's visible example uses 0.05. The final ranking can depend on ridge. |
| Reference ATE | Independent Monte Carlo integral of known response | The empirical reference comes from observed eight-arm platform data. Monte Carlo SE is saved. |
| MAPE set | `001`, `100`, `111`, `101`, `011` | Fixed to combinations with reported APE in the paper's Table 2. Synthetic significance p-values are diagnostic only. |
| Initialization | Common Glorot arrays; output scale from training outcomes | Paired-backend choice. The original empirical random state is unavailable. |
| Random seeds | Reference 2718; user data and training 10234 | Fixed project seeds; no paper claim. The response coefficients are saved directly and require no draw seed. |

The generator is fully specified by `configs/figure7.json`, `feature_schema.json`, and `response_parameters.json`. The preprocessing is one-hot expansion, not a learned embedding layer. Categories are intentionally uninformative. The input-feature comparison therefore isolates encoding and input width, not the platform's predictive categorical structure. The fixed response coefficients were chosen to give nontrivial treatment effects and a best combination of `101`; they were not fitted to the paper's curve.

The visible notebook has differences from the method used here: its CSV has all eight treatment combinations, whereas this experiment fits only five; its Figure 7 plotting example uses a shorter run and ridge 0.05. In the notebook's score calculation, a target-treatment derivative appears alongside an observed-treatment residual; here the loss gradient is evaluated entirely at observed treatment, in line with the paper's formula. These observations concern the public notebook path and do not establish an error in the authors' private empirical analysis.

Figure 7's left-axis label is retained from the paper, whose Section 5.3 calls cross-validation error “training error” for simplicity. MSE and MAPE have different units and separate axes; their vertical positions cannot be compared directly. The simulation also differs in scale, sample size, outcome noise, treatment effects, and first-stage fitting, so matching line heights or fluctuations would not demonstrate a faithful data replacement.

TensorFlow and PyTorch share data, initialization arrays, and the score implementation but use their native Adam optimizers. Matching numeric hyperparameters does not force bitwise-identical updates. Agreement across backends is a check on the training implementation, not independent validation of the shared statistical formula.
