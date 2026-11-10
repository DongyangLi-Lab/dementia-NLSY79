"""Probe TabICL's memory-dependent inference path against the saved baseline."""
import os
import runpy
import sys
from pathlib import Path
from tabicl.model.inference import InferenceManager

cap = float(os.environ['TABICL_PROBE_MEMORY_MB'])
original_memory = InferenceManager.get_available_gpu_memory
original_estimate = InferenceManager.estimate_safe_batch_size
original_call = InferenceManager.__call__

def memory(self):
    return min(original_memory(self), cap)

def estimate(self, *args, **kwargs):
    result = original_estimate(self, *args, **kwargs)
    print(f'PATH estimate enc={self.enc_name} available_mb={result[0]:.1f} batch={result[1]}', flush=True)
    return result

def call(self, forward_fn, inputs, *args, **kwargs):
    first = next(iter(inputs.values()))
    result = original_call(self, forward_fn, inputs, *args, **kwargs)
    print(f'PATH output enc={self.enc_name} shape={tuple(first.shape)} input={first.dtype} output={result.dtype} offload={self.offload}', flush=True)
    return result

InferenceManager.get_available_gpu_memory = memory
InferenceManager.estimate_safe_batch_size = estimate
InferenceManager.__call__ = call
sys.argv = ['artifacts/tabicl_local_permutation_shard.py', '--worker', '-1']
sys.path.insert(0, str(Path('artifacts').resolve()))
sys.path.insert(0, str(Path('.').resolve()))
runpy.run_path(sys.argv[0], run_name='__main__')
