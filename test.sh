# 这个脚本做这个事情，首先创造一个变量保存现在的路径，然后进入`/root/autodl-fs/Diffusion-based-Fluid-Super-resolution/Example`，最后再返回到原来的路径

# 保存当前工作目录
orig_dir=$(pwd)

clear

cd /root/autodl-fs/Diffusion-based-Fluid-Super-resolution/Example
python main.py --config kmflow_re1000_rs256_sparse_recons_conditional.yml --seed 1234 --sample_step 1 --t 240 --r 30

# 返回到原来的路径
cd "$orig_dir"