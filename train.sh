orig_dir=$(pwd)

clear

cd /root/autodl-fs/Diffusion-based-Fluid-Super-resolution/Example/train_ddpm
export CUDA_VISIBLE_DEVICES=0;
python main.py --config ./km_re1000_rs256_conditional.yml --exp ./experiments/km256/ --doc ./weights/km256/ --ni
cd "$orig_dir"