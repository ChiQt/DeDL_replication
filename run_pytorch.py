"""Train the PyTorch model and reproduce the synthetic Figure 7.

Usage: python run_pytorch.py
       python run_pytorch.py --output-dir outputs/pytorch
"""
from dedl.runtime import configure_cpu

configure_cpu()

from dedl.runner import run_cli
from dedl.torch_model import TorchModel


if __name__ == "__main__":
    run_cli("pytorch", TorchModel)
