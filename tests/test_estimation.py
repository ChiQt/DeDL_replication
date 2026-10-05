"""Checks for the score formula and the separation of experimental data roles."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dedl.data import ROOT, prepare_dataset, read_config, conditional_means, structural_parameters
from dedl.estimators import check_formula


class EstimationTests(unittest.TestCase):
    def test_gradient_and_orthogonality_identities(self):
        # Includes finite differences, zero-residual correction, and a local
        # orthogonality check with no ridge; positive ridge is a separate choice.
        self.assertTrue(check_formula().startswith("PASS"))

    def test_response_uses_fixed_continuous_features(self):
        config = read_config(ROOT / "configs/figure7.json")
        matrix, scale = structural_parameters(config)
        x = np.zeros((3, config["d"]))
        x[1, 10:] = 1
        x[2, 0] = 0.25
        means = conditional_means(x, matrix, scale)
        np.testing.assert_allclose(means[0], means[1])
        self.assertFalse(np.allclose(means[0], means[2]))

    def test_support_splits_cache_and_feature_encoding(self):
        config = read_config(ROOT / "configs/figure7.json")
        config.update(n=400, mc_n=2000, reference_n_per_group=400)
        with tempfile.TemporaryDirectory() as temporary:
            base = prepare_dataset(config, temporary, "base", 10234)
            with np.load(base / "synthetic_data.npz") as data:
                self.assertEqual(set(data["treatment_index"]), set(range(5)))
                self.assertEqual(data["x"].shape, (400, 87))
                np.testing.assert_array_equal(data["x"][:, :10], data["continuous"])
                self.assertEqual(data["categorical_codes"].shape, (400, 16))
                np.testing.assert_array_equal(data["mape_mask"],
                                              [False, True, False, True, True, False, True, True])
                for fold in range(4):
                    with np.load(base / ("split_" + str(fold) + ".npz")) as split:
                        ids = np.concatenate([split["train"], split["validation"], split["score"]])
                        np.testing.assert_array_equal(np.sort(ids), np.arange(config["n"]))
            self.assertEqual(prepare_dataset(config, temporary, "base", 10234), base)
            changed = dict(config, n=401)
            with self.assertRaises(ValueError):
                prepare_dataset(changed, temporary, "base", 10234)


if __name__ == "__main__":
    unittest.main()
