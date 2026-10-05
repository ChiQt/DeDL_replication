"""Generate known-truth outcomes and isolated train/validation/score splits."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCHEMA = json.loads((ROOT / "feature_schema.json").read_text(encoding="utf-8"))
CONTINUOUS_DIMENSION = len(FEATURE_SCHEMA["continuous"])
CATEGORICAL_WIDTHS = [int(item["levels"]) for item in FEATURE_SCHEMA["categorical"]]
ENCODED_DIMENSION = CONTINUOUS_DIMENSION + sum(CATEGORICAL_WIDTHS)
COMBOS = np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,1,1],
                   [1,1,0], [1,0,1], [0,1,1]], dtype=np.float64)
OBSERVED = COMBOS[:5]


def config_hash(config):
    content = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config["m"] != 3 or CONTINUOUS_DIMENSION != 10
            or len(CATEGORICAL_WIDTHS) != 16 or config["d"] != ENCODED_DIMENSION):
        raise ValueError("Expected three treatments, ten continuous and sixteen categorical variables.")
    if config["n"] < 100 or config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Invalid sample count, epoch count, or batch size.")
    if not 0 < config["reference_alpha"] < 1 or config["ridge"] <= 0:
        raise ValueError("Use a significance level in (0, 1) and positive ridge.")
    return config


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structural_parameters(plan):
    """Load the fixed coefficients of the synthetic response model."""
    response = json.loads((ROOT / "response_parameters.json").read_text(encoding="utf-8"))
    matrix = np.asarray(response["coefficient_matrix"], dtype=np.float64)
    if matrix.shape != (4, 6):
        raise ValueError("Expected four coefficient rows and six fixed basis functions.")
    scale = float(response["scale"])
    return matrix, scale


def nuisance_truth(x, matrix, scale):
    features = np.asarray(x, dtype=np.float64)
    basis = np.column_stack([
        np.ones(len(features)), np.sin(2 * np.pi * features[:, 0]),
        features[:, 1] - .5, features[:, 2] - .5,
        features[:, 3] - .5, features[:, 4] - .5,
    ])
    coefficients = basis @ matrix.T
    return np.column_stack([coefficients, np.full(len(x), scale)])


def conditional_means(x, matrix, scale, scenario="base"):
    theta = nuisance_truth(x, matrix, scale)
    index = theta[:, [0]] + theta[:, 1:4] @ COMBOS.T
    means = scale / (1 + np.exp(-np.clip(index, -500, 500)))
    if scenario == "mild_misspec":
        response = json.loads((ROOT / "response_parameters.json").read_text(encoding="utf-8"))
        interaction = COMBOS[:, 1] * COMBOS[:, 2]
        means += (float(response["mild_misspec_delta"])
                  * (1 + .4 * (np.asarray(x)[:, [5]] - .5)) * interaction)
    elif scenario != "base":
        raise ValueError(f"Unknown scenario: {scenario}")
    return means


def generate_reference(plan, matrix, scale, halfwidth, scenario):
    """Compute population ATE and record synthetic significance diagnostics.

    The large Monte Carlo sample integrates known conditional means. Separate
    independent treatment groups produce Welch p-values for inspection only.
    The MAPE support is fixed from the paper's Table 2, never selected by these
    synthetic p-values. Neither reference sample updates the neural network.
    """
    rng = np.random.default_rng(plan["reference_seed"])
    # The response function depends on the continuous variables only.
    mc_x = rng.uniform(0, 1, (plan["mc_n"], CONTINUOUS_DIMENSION))
    means = conditional_means(mc_x, matrix, scale, scenario)
    contrasts = means - means[:, [0]]
    truth = contrasts.mean(axis=0)
    mc_se = contrasts.std(axis=0, ddof=1) / np.sqrt(len(mc_x))

    experiment_rng = np.random.default_rng(plan["reference_seed"] + 1000)
    reference_outcomes = []
    for group in range(8):
        x = experiment_rng.uniform(0, 1, (plan["reference_n_per_group"], CONTINUOUS_DIMENSION))
        noise = experiment_rng.uniform(-1, 1, len(x)) * halfwidth
        reference_outcomes.append(conditional_means(x, matrix, scale, scenario)[:, group] + noise)

    empirical_ate = np.array([values.mean() for values in reference_outcomes])
    empirical_ate -= empirical_ate[0]
    pvalues = np.ones(8)
    for group in range(1, 8):
        pvalues[group] = ttest_ind(
            reference_outcomes[group], reference_outcomes[0], equal_var=False
        ).pvalue
    # Figure 7 evaluates the groups significant in the private real experiment.
    # Fix that public Table 2 set here; synthetic Welch p-values are diagnostic.
    mask = np.array([False, True, False, True, True, False, True, True])
    return truth, mc_se, empirical_ate, pvalues, mask


def generate_one(plan, scenario, seed, matrix, scale, data_root):
    destination = Path(data_root) / scenario / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "synthetic_data.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {path}")

    rng = np.random.default_rng(seed)
    continuous = rng.uniform(0, 1, (plan["n"], CONTINUOUS_DIMENSION)).astype(np.float32)
    treatment_index = rng.integers(0, 5, len(continuous))
    noise = rng.uniform(-1, 1, len(continuous)) * plan["scenarios"][scenario]
    means = conditional_means(continuous, matrix, scale, scenario)
    y = (means[np.arange(len(continuous)), treatment_index] + noise).astype(np.float32)

    fold_id = np.empty(len(continuous), dtype=int)
    for group in range(5):
        users = np.flatnonzero(treatment_index == group)
        rng.shuffle(users)
        fold_id[users] = np.arange(len(users)) % 4

    # Separate RNG keeps the continuous data, assignments, noise and folds fixed.
    categorical_rng = np.random.default_rng(seed + 40000)
    categorical_codes = np.column_stack([
        categorical_rng.integers(0, width, len(continuous))
        for width in CATEGORICAL_WIDTHS
    ]).astype(np.int16)
    one_hot = [np.eye(width, dtype=np.float32)[categorical_codes[:, j]]
               for j, width in enumerate(CATEGORICAL_WIDTHS)]
    x = np.column_stack([continuous, *one_hot]).astype(np.float32)
    assert x.shape == (plan["n"], ENCODED_DIMENSION)

    truth, mc_se, empirical, pvalues, mask = generate_reference(
        plan, matrix, scale, plan["scenarios"][scenario], scenario
    )
    np.savez_compressed(
        path, x=x, continuous=continuous, categorical_codes=categorical_codes,
        y=y, treatment_index=treatment_index, fold=fold_id,
        matrix=matrix, true_scale=scale, truth=truth, truth_mc_se=mc_se,
        reference_ate=empirical, reference_pvalue=pvalues, mape_mask=mask
    )
    for fold in range(4):
        split_rng = np.random.default_rng(seed + 20000 + fold)
        score_ids = np.flatnonzero(fold_id == fold)
        pool = np.flatnonzero(fold_id != fold)
        split_rng.shuffle(pool)
        validation_count = len(pool) // 10
        np.savez_compressed(
            destination / f"split_{fold}.npz",
            train=pool[validation_count:], validation=pool[:validation_count], score=score_ids
        )

    config = {
        "scenario": scenario, "data_seed": seed, "n": len(x), "dimension": x.shape[1],
        "raw_continuous_count": CONTINUOUS_DIMENSION,
        "raw_categorical_count": len(CATEGORICAL_WIDTHS),
        "one_hot_count": sum(CATEGORICAL_WIDTHS),
        "feature_schema_sha256": file_hash(ROOT / "feature_schema.json"),
        "matrix": matrix.tolist(), "true_scale": scale,
        "response_sha256": file_hash(ROOT / "response_parameters.json"),
        "misspecification": scenario == "mild_misspec",
        "mape_mask_source": "paper Table 2 significant non-control groups",
        "noise_halfwidth": plan["scenarios"][scenario],
        "noise_variance": plan["scenarios"][scenario] ** 2 / 3,
        "outcome_rescaling": 1.0, "true_ates": truth.tolist(),
        "reference_ates": empirical.tolist(), "reference_pvalues": pvalues.tolist(),
        "mape_groups": ["".join(map(str, row.astype(int))) for row in COMBOS[mask]],
        "observed_group_counts": np.bincount(treatment_index, minlength=5).tolist(),
        "best_true_combination": "".join(map(str, COMBOS[np.argmax(truth)].astype(int))),
        "data_sha256": file_hash(path), "plan_sha256": config_hash(plan)
    }
    (destination / "data_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Prepared {scenario}, seed {seed}; true ATE={np.round(truth, 4)}", flush=True)


def prepare_dataset(config, data_root, scenario, seed):
    """Reuse verified inputs, or generate one dataset under an exclusive lock."""
    folder = Path(data_root) / scenario / ("seed_" + str(seed))
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "data_config.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["plan_sha256"] != config_hash(config):
            raise ValueError("Configuration changed; select a new --data-dir.")
        if saved["response_sha256"] != file_hash(ROOT / "response_parameters.json"):
            raise ValueError("Response parameters changed; select a new --data-dir.")
        if saved["feature_schema_sha256"] != file_hash(ROOT / "feature_schema.json"):
            raise ValueError("Feature schema changed; select a new --data-dir.")
        if saved["data_sha256"] != file_hash(folder / "synthetic_data.npz"):
            raise ValueError("Cached dataset hash does not match its metadata.")
        return folder
    lock = folder / ".preparing"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Dataset preparation already started. Retry when it finishes; "
                           "remove a stale .preparing file only after checking the process.")
    os.close(descriptor)
    try:
        matrix, scale = structural_parameters(config)
        generate_one(config, scenario, seed, matrix, scale, data_root)
    finally:
        lock.unlink()
    return folder
