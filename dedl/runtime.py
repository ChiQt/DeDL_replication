"""CPU settings used by the reference runs; apply before importing a framework."""
import os


def configure_cpu():
    # Small dense networks are inexpensive on CPU; limiting threads also makes
    # independent runs suitable for a shared workstation or compute server.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "TF_NUM_INTRAOP_THREADS"]:
        os.environ[name] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
