import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".npz":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def get_shapes(data):
    if isinstance(data, np.lib.npyio.NpzFile):
        return {name: arr.shape for name, arr in data.items()}
    else:
        return data.shape

def print_data_shapes(path):
    data = load_data(path)
    shapes = get_shapes(data)
    if isinstance(shapes, dict):
        print(f"Archive at {path} contains:")
        for name, shape in shapes.items():
            print(f"  {name}: {shape}")
    else:
        print(f"Array at {path} shape: {shapes}")

def visualize_data(path, n_indices=None, t_indices=None):
    data = load_data(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        first_key = list(data.keys())[0]
        arr = data[first_key]
    else:
        arr = data

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # 设置图表背景透明
    fig.patch.set_alpha(0)
    samples_to_show = 4

    if arr.ndim == 4:
        if n_indices is None:
            n_indices = random.sample(range(arr.shape[0]), min(samples_to_show, arr.shape[0]))
        im = None
        for i, idx in enumerate(n_indices):
            if t_indices is None:
                t_idx = random.randint(0, arr.shape[1] - 1)
            else:
                t_idx = t_indices[i]
            slice_2d = arr[idx, t_idx]
            ax = axes[i]
            im = ax.imshow(slice_2d, cmap='twilight', interpolation='none', vmin=-23, vmax=23)
            ax.axis('off')
            # 设置子图背景透明
            ax.patch.set_alpha(0)
            ax.set_title(f"Process={idx}, Time={t_idx}")
    elif arr.ndim == 3:
        if t_indices is None:
            t_indices = random.sample(range(arr.shape[0]), min(samples_to_show, arr.shape[0]))
        im = None
        for i, t_idx in enumerate(t_indices):
            slice_2d = arr[t_idx]
            ax = axes[i]
            im = ax.imshow(slice_2d, cmap='twilight', interpolation='none', vmin=-23, vmax=23)
            ax.axis('off')
            # 设置子图背景透明
            ax.patch.set_alpha(0)
            ax.set_title(f"Time={t_idx}")
    else:
        print(f"无法可视化 {path}，数据维度过低或不符合预期：{arr.shape}")
        return

    out_pdf = os.path.splitext(path)[0] + '_multi_frames.pdf'
    out_png = os.path.splitext(path)[0] + '_multi_frames.png'
    out_eps = os.path.splitext(path)[0] + '_multi_frames.eps'
    # 添加透明背景参数
    plt.savefig(out_pdf, bbox_inches='tight', transparent=True)
    plt.savefig(out_png, bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(out_eps, bbox_inches='tight', transparent=True)
    plt.close()

def main():
    high_res_path = "Example/data/kf_2d_re1000_256_40seed.npy"
    low_res_path = "Example/data/kmflow_sampled_data_irregnew.npz"

    # 载入高分辨率数据，抽取相同的随机过程和时间帧
    data_high = load_data(high_res_path)
    if isinstance(data_high, np.lib.npyio.NpzFile):
        arr_high = data_high[list(data_high.keys())[0]]
    else:
        arr_high = data_high
    
    samples_to_show = 4
    # 若为 (N, T, H, W)，随机过程和时间帧
    if arr_high.ndim == 4:
        n_indices = random.sample(range(arr_high.shape[0]), min(samples_to_show, arr_high.shape[0]))
        t_indices = [random.randint(0, arr_high.shape[1] - 1) for _ in range(len(n_indices))]
    elif arr_high.ndim == 3:
        n_indices, t_indices = None, random.sample(range(arr_high.shape[0]), min(samples_to_show, arr_high.shape[0]))
    else:
        n_indices, t_indices = None, None

    print_data_shapes(high_res_path)
    visualize_data(high_res_path, n_indices=n_indices, t_indices=t_indices)

    print_data_shapes(low_res_path)
    visualize_data(low_res_path, n_indices=n_indices, t_indices=t_indices)

if __name__ == "__main__":
    main()