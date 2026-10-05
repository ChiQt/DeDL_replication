"""Redraw a saved curve without importing TensorFlow or PyTorch."""
import argparse
from pathlib import Path

from dedl.plotting import figure7, load_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True, help="A fold_0_curve.csv or aggregate_curve.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path; a PNG is also saved.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure7(load_rows(args.curve), args.output,
            "Synthetic Figure 7, redrawn from saved measurements: " + args.curve.name)
    print(args.output)
