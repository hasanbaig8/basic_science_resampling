import sys
import os
sys.path.append('/workspace/basic_science_resampling')
from pipeline.store_activations import store_activations

store_activations(
    model_name='Qwen/Qwen3-8b',
    rollouts_path='/workspace/basic_science_resampling/data/train_rollouts.json',
    activations_dir='/workspace/basic_science_resampling/data/activations/train/',
)

'''store_activations(
    model_name='Qwen/Qwen3-8b',
    rollouts_path='/workspace/basic_science_resampling/data/test_rollouts.json',
    activations_dir='/workspace/basic_science_resampling/data/activations/test/',
)'''