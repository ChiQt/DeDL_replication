"""Train the TensorFlow model and reproduce the synthetic Figure 7.

Usage: python run_tensorflow.py
       python run_tensorflow.py --output-dir outputs/tensorflow
"""
from dedl.runtime import configure_cpu

configure_cpu()

from dedl.runner import run_cli
from dedl.tf_model import TensorFlowModel


if __name__ == "__main__":
    run_cli("tensorflow", TensorFlowModel)
