# Code guide / 代码解读

## Data flow

`run_tensorflow.py` and `run_pytorch.py` select independent network classes, then call the shared experiment runner. Only the selected deep-learning framework is imported by each script. `dedl/data.py` generates inputs and references; `dedl/estimators.py` computes the same SDL, DeDL, and LR estimands for both backends.

```text
run_tensorflow.py ── TensorFlowModel ─┐
                                     ├── runner.run() ── curves, ATE tables, figures
run_pytorch.py ───── TorchModel ──────┘           │
                                   data.prepare_dataset()
                                   estimators.influence_scores()
                                   plotting.figure7()
```

The shared estimator makes backend comparisons interpretable, but agreement between backends does not independently prove the score formula.

## Synthetic data / 数据生成

`response_parameters.json` fixes a coefficient matrix $B$ and scale $c=20$. The first ten features are independent $U(0,1)$ draws. A six-dimensional basis uses an intercept, $\sin(2\pi x_0)$, and centered $x_1,\ldots,x_4$. The four rows of $B$ give $a(x),b_1(x),b_2(x),b_3(x)$. The conditional mean is

$$
G(x,t)=c\,\sigma\!\left(a(x)+\sum_{j=1}^3 b_j(x)t_j\right),
\qquad Y=G(X,T)+\varepsilon,\quad \varepsilon\sim U(-0.5,0.5).
$$

Sixteen additional categorical variables are sampled independently and encoded into 77 one-hot columns. `feature_schema.json` fixes their level counts. These columns do not enter $G$; they change the input representation and fitting difficulty while retaining the same response mechanism. The model therefore receives 87 columns. Treatment is sampled uniformly among `000`, `001`, `010`, `100`, and `111`. The means of the three other combinations are available to the simulator for evaluation, but their outcomes do not enter training or DeDL's observed-data score.

`generate_reference()` integrates known conditional means over 500,000 independently sampled continuous-feature vectors to approximate population ATEs. A separate 2,000-users-per-arm reference is saved as a diagnostic. The MAPE group set is fixed from the paper's Table 2 (`001`, `100`, `111`, `101`, `011`), independently of p-values computed on those reference users.

`generate_one()` writes `synthetic_data.npz` and four disjoint `split_*.npz` files. `prepare_dataset()` checks hashes before reusing a cached dataset. True parameters and reference ATEs are read only in diagnostics and evaluation, never for optimization.

## DNN and training / 网络与训练

The TensorFlow builder adapts the authors' public `sig_dnn` architecture. PyTorch implements the same two-branch layout. Each branch has three dense 20-unit ReLU layers. One branch produces $a(x)$ and the other produces $b(x)\in\mathbb R^3$; a shared trainable scalar $c$ closes the sigmoid link. `initialization.py` gives both backends the same starting dense-layer arrays. The scale starts at 1.1 times the training-outcome 99th percentile.

For each of four scoring folds, `runner.py` trains on 90% of the remaining three folds and evaluates prediction MSE on the other 10%. Every five epochs, plus epochs 0 and 1, it measures SDL and DeDL on the untouched scoring fold. LR fits on the same training users using an intercept, all encoded covariates, and three treatment indicators. It is constant across epochs.

At a fixed epoch, the four folds each estimate an ATE. `runner.py` pools the fold score sums and user counts, then computes MAPE from the pooled ATE. Therefore `aggregate_curve.csv` is not an average of four fold MAPE curves. The figure calls validation MSE “Training MSE”, following the paper's Section 5.3 terminology. Actual in-sample MSE is saved in the CSV separately.

## Estimators / 估计量

For $t_0=000$, SDL averages the fitted contrast

$$
\widehat H_i(t)=\widehat G(X_i,t)-\widehat G(X_i,t_0).
$$

DeDL uses the five response coordinates $\theta=(a,b_1,b_2,b_3,c)$, not all DNN weights. The loss gradient uses the **observed** $T_i$:

$$
\ell_{\theta,i}=2\nabla_\theta\widehat G(X_i,T_i)
\bigl(\widehat G(X_i,T_i)-Y_i\bigr),
\qquad
\widehat\Lambda_i=\frac25\sum_{s\in\mathcal T_o}
\nabla_\theta\widehat G(X_i,s)\nabla_\theta\widehat G(X_i,s)^\top.
$$

$$
\widehat\psi_i(t)=\widehat H_i(t)-
\nabla_\theta\widehat H_i(t)^\top
(\widehat\Lambda_i+\lambda I)^{-1}\ell_{\theta,i},
\qquad \lambda=0.0005.
$$

`estimators.py` solves the linear system rather than forming an explicit inverse. The outer-product $\widehat\Lambda$ is a plug-in for the expected Hessian at the true conditional mean; it is not the full Hessian at arbitrary fitted parameters. A second ridge value, 0.05, is calculated at the same checkpoints for diagnosis and saved in `dedl_ridge_005_mape`. Ridge does not enter DNN training.

MAPE averages percentage error over the fixed five-group set. MAE and RMSE average over all seven nonbaseline effects. The `oracle_nuisance.csv` diagnostic substitutes known conditional-mean parameters into the score but still uses a finite scoring sample; it is not a lower bound on measured error.

## Outputs and checks / 输出与检查

| File under `outputs/<backend>/` | Meaning |
|---|---|
| `figure7_fourfold.pdf` / `.png` | Primary, four-fold aggregate curve |
| `aggregate_curve.csv` | Aggregate measurements at each checkpoint |
| `fold_0_curve.csv` … `fold_3_curve.csv` | Individual held-out fold measurements |
| `fold_final_ates.csv`, `final_ates.csv` | End-of-training treatment-level estimates |
| `figure7_synthetic.pdf` / `.png` | First-fold diagnostic figure |
| `oracle_nuisance.csv` | Known-response diagnostic |
| `model_fold_*.h5` / `.pt` | Backend-specific final weights |
| `config.json`, `run_summary.json` | Config, hashes, software versions, elapsed time |

Generated data are under `data/synthetic/base/seed_10234/`. A completed output directory is protected from overwrite. After changing the generator or config, use a new `--data-dir` and `--output-dir`. `python -m unittest discover -s tests -v` checks response calculations, treatment support, splits, and score identities. [DIFFERENCES.md](DIFFERENCES.md) traces settings to the paper or to this simulation.
