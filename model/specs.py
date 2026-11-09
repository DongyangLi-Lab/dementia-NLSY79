"""Part 1 model families and validation grids."""
import glob
import hashlib
import os
from functools import lru_cache
from pathlib import Path

MODELS = ("LR", "GNB", "DT", "LightGBM", "CatBoost", "RealMLP", "TabICL", "GAT")
GPU_MODELS = {"LightGBM", "CatBoost", "RealMLP", "TabICL", "GAT"}

@lru_cache(maxsize=1)
def model_specs():
    """name -> (class, hyperparameter grid, fixed arguments). Grids follow Table 1 of the manuscript."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from pytabkit import RealMLP_TD_Classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from tabicl import TabICLClassifier
    from model.neuralclassifier_fixed import GATClassifier
    return {
        'LR': (LogisticRegression, {'C': [1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']},
               {'random_state': 42}),
        'GNB': (GaussianNB, {'var_smoothing': [1e-10, 1e-7, 1e-4], 'priors': [None, [0.5, 0.5], [0.4, 0.6]]}, {}),
        'DT': (DecisionTreeClassifier, {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 4]}, {'random_state': 42}),
        'LightGBM': (LGBMClassifier, {'num_leaves': [20, 40], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.05, 0.1]},
                     {'verbose': -1, 'random_state': 42}),
        'CatBoost': (CatBoostClassifier, {'iterations': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8]},
                     {'verbose': 0, 'random_state': 42, 'allow_writing_files': False}),
        'GAT': (GATClassifier, {'learning_rate': [1e-4, 1e-2], 'dropout_p': [0.1, 0.5]},
                {'hidden_layer_sizes': (128, 64, 32), 'max_iter': 50, 'use_residual': True, 'use_batch_norm': False,
                 'batch_size': 256, 'random_state': 42, 'cuda': True, 'reference_size': 255, 'standardize': False,
                 'max_restarts': 3}),
        'RealMLP': (RealMLP_TD_Classifier, {'batch_size': [128, 256], 'n_epochs': [50, 100]}, {'random_state': 42}),
        'TabICL': (TabICLClassifier, {'n_estimators': [16, 32], 'batch_size': [8, 16]},
                   {'use_amp': True, 'checkpoint_version': 'tabicl-classifier-v1.1-0506.ckpt', 'random_state': 42}),
    }


def _tabicl_checkpoint(version):
    """Path and SHA-256 of the cached TabICL checkpoint file actually used (Hugging Face cache)."""
    import glob
    paths = glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', '*TabICL*', 'snapshots',
                                   '*', version))
    if not paths:
        return {'version': version, 'path': None, 'sha256': None}
    digest = hashlib.sha256(open(paths[0], 'rb').read()).hexdigest()
    rel = paths[0].split(os.path.join('huggingface', 'hub') + os.sep)[-1].replace(os.sep, '/')
    return {'version': version, 'path': f'huggingface/hub/{rel}', 'sha256': digest}



def configuration(model, device, threads):
    cls, grid, defaults = model_specs()[model]
    fixed = defaults.copy()
    if model in GPU_MODELS and device == "cpu":
        raise ValueError(f"{model} requires a GPU in the revised pipeline")
    if model == "LightGBM":
        # OpenCL device indices are separate from CUDA visibility.
        fixed.update(device_type="gpu", gpu_platform_id=int(os.environ.get("HRS_OPENCL_PLATFORM", "0")),
                     gpu_device_id=int(os.environ.get("HRS_OPENCL_DEVICE", "-1")), n_jobs=threads)
    elif model == "CatBoost":
        fixed.update(task_type="GPU", devices="0", thread_count=threads)
    elif model == "RealMLP":
        fixed.update(device="cuda:0", n_threads=threads)
    elif model == "TabICL":
        fixed.update(device="cuda:0", n_jobs=threads,
                     inference_config={component: {"safety_factor": 0.65}
                                       for component in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")})
    return cls, grid, fixed
