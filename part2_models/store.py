"""Immutable run stages and checksum-verified model persistence."""
import importlib.metadata
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from .data import ROOT, require, sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def environment():
    versions = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'catboost', 'lightgbm']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_hashes():
    files = list((ROOT / 'part2_models').glob('*.py')) + [ROOT/'stage2_config.json', ROOT/'metadata/source_rules.json']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def verify_stage(directory):
    directory = Path(directory)
    manifest = json.loads((directory / '_SUCCESS.json').read_text())
    for name, digest in manifest['files'].items():
        require(sha256(directory / name) == digest, 'Changed run artifact: ' + name)
    return manifest


def start_stage(directory, contract):
    directory = Path(directory)
    contract = dict(contract, code=code_hashes(), environment=environment())
    if directory.exists():
        require((directory/'_SUCCESS.json').is_file(), 'Incomplete stage exists; preserve it and use a new output directory')
        previous = verify_stage(directory)
        require(previous['contract'] == contract, 'Existing stage has a different data/configuration/code/environment contract')
        print('Reused verified stage: ' + str(directory), flush=True)
        return False
    directory.mkdir(parents=True)
    write_json(directory/'contract.json', contract)
    return True


def finish_stage(directory):
    directory = Path(directory)
    manifest = {'contract': json.loads((directory/'contract.json').read_text()),
                'files': {p.relative_to(directory).as_posix(): sha256(p)
                          for p in sorted(directory.rglob('*')) if p.is_file() and p.name != '_SUCCESS.json'}}
    write_json(directory/'_SUCCESS.json', manifest)


def save_model(model, directory, metadata, replay_frame):
    directory = Path(directory)
    require(not directory.exists(), 'Model destination already exists')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temporary:
        pending = Path(temporary)/'model'
        pending.mkdir()
        joblib.dump(model, pending/'pipeline.joblib', compress=3)
        loaded = joblib.load(pending/'pipeline.joblib')
        error = float(np.max(np.abs(model.predict(replay_frame) - loaded.predict(replay_frame))))
        require(error <= 1e-10, 'Saved-model replay differs')
        info = dict(metadata, features=model.features, learner=model.learner, params=model.params,
                    environment=environment(), replay_max_abs_difference=error,
                    pipeline_sha256=sha256(pending/'pipeline.joblib'))
        if model.learner == 'catboost':
            model.model.save_model(str(pending/'regressor.cbm'))
            info['native_model_sha256'] = sha256(pending/'regressor.cbm')
        elif model.learner == 'lightgbm':
            model.model.booster_.save_model(str(pending/'regressor.txt'))
            info['native_model_sha256'] = sha256(pending/'regressor.txt')
        write_json(pending/'metadata.json', info)
        pending.rename(directory)


def load_model(directory):
    """Load artifacts produced by this project in the recorded Python environment."""
    directory = Path(directory)
    info = json.loads((directory/'metadata.json').read_text())
    require(sha256(directory/'pipeline.joblib') == info['pipeline_sha256'], 'Model checksum mismatch')
    current = environment()
    for name in ['python', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib'] + ([info['learner']] if info['learner'] in ['catboost', 'lightgbm'] else []):
        require(current[name] == info['environment'][name], 'Model environment mismatch: ' + name)
    model = joblib.load(directory/'pipeline.joblib')
    require(model.features == info['features'], 'Saved feature order differs')
    return model, info
