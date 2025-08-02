echo "ckpt 250"
CUDA_VISIBLE_DEVICES=1 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-24-1603.25/checkpoint-250"
sleep 10

echo "ckpt 500"
CUDA_VISIBLE_DEVICES=1 python3 exp1.py --ckpt_path "/fs/archive/share/ft_2025-07-24-1603.25/checkpoint-500"
sleep 10

echo "ckpt 1000"
CUDA_VISIBLE_DEVICES=1 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-24-1603.25/checkpoint-1000"
sleep 10

echo "ckpt 2000"
CUDA_VISIBLE_DEVICES=1 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-25-0626.09/checkpoint-2000"
sleep 10