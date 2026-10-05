"""Training, held-out estimation, and output shared by the two entry scripts."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from .data import ROOT, COMBOS, file_hash, config_hash, read_config, prepare_dataset, nuisance_truth
from .estimators import check_formula, influence_scores, metrics
from .initialization import initial_weights
from .plotting import figure7


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(epoch, mse, train_mse, estimates, truth, mask):
    row = {"epoch": epoch, "validation_mse": mse, "in_sample_training_mse": train_mse}
    for method, estimate in estimates.items():
        row.update({method + "_" + name: value
                    for name, value in metrics(estimate, truth, mask).items()})
    return row


def run_cli(backend, model_class):
    """The entry script supplies its own framework; no second backend is imported."""
    parser = argparse.ArgumentParser(description="Reproduce synthetic Figure 7 with " + backend)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/figure7.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / backend)
    parser.add_argument("--scenario", choices=["base", "mild_misspec"], default="base")
    parser.add_argument("--seed", type=int, default=10234)
    parser.add_argument("--folds", type=int, choices=[1, 4], default=4)
    args = parser.parse_args()
    args.backend = backend
    run(args, model_class)


def run(args, model_class):
    start = time.time()
    plan = read_config(args.config)
    checks = check_formula()
    print(checks, flush=True)
    data_directory = prepare_dataset(plan, args.data_dir, args.scenario, args.seed)
    dataset_path = data_directory / "synthetic_data.npz"
    dataset = np.load(dataset_path)
    x, y = dataset["x"], dataset["y"]
    treatment_index = dataset["treatment_index"]
    treatment = COMBOS[treatment_index].astype(np.float32)
    truth, mask = dataset["truth"], dataset["mape_mask"]
    assert np.max(treatment_index) < 5, "Hidden treatments leaked into the estimation data."
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_summary.json").exists():
        raise FileExistsError(f"Completed run exists; preserve it: {out}")

    checkpoints = sorted(set([0, 1, plan["epochs"]] + list(range(plan["every"], plan["epochs"] + 1, plan["every"]))))
    (out / "config.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    totals = {epoch: {"sdl": np.zeros(8), "dedl": np.zeros(8), "ridge_alt": np.zeros(8),
                      "n": 0, "validation_sse": 0.0, "validation_n": 0,
                      "training_sse": 0.0, "training_n": 0} for epoch in checkpoints}
    lr_total, lr_count = np.zeros(8), 0
    all_initial_predictions, fold_final_rows, oracle_rows = [], [], []
    final_scores = []

    for fold in range(args.folds):
        split = np.load(data_directory / f"split_{fold}.npz")
        train_ids, validation_ids, score_ids = split["train"], split["validation"], split["score"]
        assert not np.intersect1d(train_ids, validation_ids).size
        assert not np.intersect1d(train_ids, score_ids).size
        assert not np.intersect1d(validation_ids, score_ids).size
        initial_scale = float(np.quantile(y[train_ids], 0.99) * 1.1)
        model = model_class(x.shape[1], initial_weights(args.seed + fold, x.shape[1]),
                            initial_scale, plan["learning_rate"])
        model.bind_data(x, treatment, y)
        all_initial_predictions.append(model.predict(x[:64], treatment[:64]))
        shuffle_rng = np.random.default_rng(args.seed + 30000 + fold)

        # Use exactly the same training users for LR and the neural network.
        design = np.column_stack([np.ones(len(train_ids)), x[train_ids], treatment[train_ids]])
        linear_coefficients = np.linalg.lstsq(design, y[train_ids], rcond=None)[0]
        lr_estimate = COMBOS @ linear_coefficients[-3:]
        lr_total += lr_estimate * len(score_ids)
        lr_count += len(score_ids)

        # A true-nuisance benchmark exists only when the structural link is correct.
        # In the misspecified scenario theta is merely a structural component.
        if args.scenario == "base":
            true_theta = nuisance_truth(x[score_ids], dataset["matrix"], float(dataset["true_scale"]))
            oracle_plugin, oracle_score, _ = influence_scores(
                true_theta, treatment_index[score_ids], y[score_ids], plan["ridge"]
            )
            oracle_rows.append({"fold": fold, **metrics(oracle_score.mean(0), truth, mask),
                                "plugin_mae": metrics(oracle_plugin.mean(0), truth, mask)["mae"]})
        else:
            oracle_rows.append({"fold": fold, "note": "not defined when the link is misspecified"})
        fold_curve = []
        for epoch in range(plan["epochs"] + 1):
            if epoch > 0:
                model.train_epoch(shuffle_rng.permutation(train_ids), plan["batch_size"])
            if epoch in totals:
                validation_residual = model.predict(x[validation_ids], treatment[validation_ids]) - y[validation_ids]
                training_residual = model.predict(x[train_ids], treatment[train_ids]) - y[train_ids]
                theta = model.parameters(x[score_ids])
                plugin, corrected, diagnostics = influence_scores(
                    theta, treatment_index[score_ids], y[score_ids], plan["ridge"]
                )
                _, alternative, _ = influence_scores(
                    theta, treatment_index[score_ids], y[score_ids], plan["diagnostic_ridge"]
                )
                estimates = {"sdl": plugin.mean(0), "dedl": corrected.mean(0), "lr": lr_estimate}
                row = metric_row(epoch, float(np.mean(validation_residual ** 2)),
                                 float(np.mean(training_residual ** 2)), estimates, truth, mask)
                row.update(diagnostics)
                row["dedl_ridge_005_mape"] = metrics(alternative.mean(0), truth, mask)["mape"]
                fold_curve.append(row)
                values = totals[epoch]
                values["sdl"] += plugin.sum(0)
                values["dedl"] += corrected.sum(0)
                values["ridge_alt"] += alternative.sum(0)
                values["n"] += len(score_ids)
                values["validation_sse"] += float(np.sum(validation_residual ** 2))
                values["validation_n"] += len(validation_ids)
                values["training_sse"] += float(np.sum(training_residual ** 2))
                values["training_n"] += len(train_ids)
                if epoch == plan["epochs"]:
                    final_scores.append(corrected)
                    for group, combination in enumerate(COMBOS.astype(int)):
                        fold_final_rows.append({
                            "fold": fold, "treatment": "".join(map(str, combination)),
                            "true_ate": truth[group], "dedl": estimates["dedl"][group],
                            "sdl": estimates["sdl"][group], "lr": lr_estimate[group]
                        })
            if epoch % 100 == 0:
                print(f"{args.backend}/{args.scenario}/{args.seed}: fold {fold+1}/{args.folds} "
                      f"epoch {epoch}; elapsed {time.time()-start:.1f}s", flush=True)

        write_csv(out / f"fold_{fold}_curve.csv", fold_curve)
        model.save(out / f"model_fold_{fold}")
        if fold == 0:
            single_fold_final = fold_curve[-1]
            figure7(fold_curve, out / "figure7_synthetic.pdf",
                    "First scoring fold of the synthetic experiment.")

    aggregate_curve = []
    for epoch in checkpoints:
        values = totals[epoch]
        estimates = {"sdl": values["sdl"] / values["n"],
                     "dedl": values["dedl"] / values["n"], "lr": lr_total / lr_count}
        row = metric_row(epoch, values["validation_sse"] / values["validation_n"],
                         values["training_sse"] / values["training_n"], estimates, truth, mask)
        row["dedl_ridge_005_mape"] = metrics(values["ridge_alt"] / values["n"], truth, mask)["mape"]
        aggregate_curve.append(row)
    write_csv(out / "aggregate_curve.csv", aggregate_curve)
    write_csv(out / "fold_final_ates.csv", fold_final_rows)
    write_csv(out / "oracle_nuisance.csv", oracle_rows)
    np.save(out / "initial_predictions.npy", np.stack(all_initial_predictions))
    if args.folds == 4:
        figure7(aggregate_curve, out / "figure7_fourfold.pdf",
                "Fourfold aggregated estimates from synthetic data.")

    score_standard_error = np.concatenate(final_scores).std(0, ddof=1) / np.sqrt(lr_count)
    write_csv(out / "final_ates.csv", [
        {"treatment": "".join(map(str, combination.astype(int))),
         "observed": bool(group < 5), "mape_included": bool(mask[group]),
         "true_ate": truth[group], "truth_mc_se": dataset["truth_mc_se"][group],
         "reference_ate": dataset["reference_ate"][group],
         "reference_pvalue": dataset["reference_pvalue"][group],
         "dedl": estimates["dedl"][group], "sdl": estimates["sdl"][group],
         "lr": estimates["lr"][group], "descriptive_score_se": score_standard_error[group]}
        for group, combination in enumerate(COMBOS)
    ])
    summary = {
        "backend": args.backend, "scenario": args.scenario, "seed": args.seed,
        "folds": args.folds, "epochs": plan["epochs"], "n": len(x),
        "elapsed_seconds": time.time() - start, "versions": model.versions(),
        "data_sha256": file_hash(dataset_path), "plan_sha256": config_hash(plan),
        "math_check": checks, "single_fold_final": single_fold_final,
        "aggregate_final": aggregate_curve[-1], "mape_mask": mask.tolist(),
        "best_true_treatment": COMBOS[np.argmax(truth)].astype(int).tolist(),
        "best_estimated_treatment": COMBOS[np.argmax(estimates["dedl"])].astype(int).tolist(),
        "notes": ["All checkpoints fixed before training; no ATE-based model selection.",
                  "Fourfold aggregate PDF is the primary figure.",
                  "Score standard errors are descriptive; no coverage guarantee is asserted."]
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
