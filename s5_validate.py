import os
import sys
sys.path.insert(0, 'dataset/model')
import numpy as np
from model import predict_batch


def load_group_data(dsp32_dir, group, n_mfcc=49, n_coef=40):
    """Load and mean-pool dsp32 data for one group"""
    dsp32_file = f"{dsp32_dir}/{group}.bin"
    dsp32_data = np.fromfile(dsp32_file, dtype=np.float32)
    count = dsp32_data.size // n_mfcc // n_coef
    data = dsp32_data.reshape((count, n_mfcc, n_coef))
    return data.mean(axis=1)


def validate_groups(dsp32_dir, groups, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for group in groups:
        data = load_group_data(dsp32_dir, group)
        probs = predict_batch(data)
        probs.astype(np.float32).tofile(os.path.join(output_dir, f"{group}.bin"))
        results[group] = {
            'mean': probs.mean(),
            'std': probs.std(),
            'max': probs.max()
        }
        print(f"{group:10s} mean={probs.mean():.9f} std={probs.std():.9f} max={probs.max():.9f}")
    return results


if __name__ == "__main__":
    dsp32_dir = "dataset/dsp32"
    groups = ["normal", "abnormal", "d1", "d2", "d3", "d4"]
    output_dir = "dataset/predict"
    validate_groups(dsp32_dir, groups, output_dir)
