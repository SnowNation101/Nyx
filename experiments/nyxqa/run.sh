echo "ckpt 5"
CUDA_VISIBLE_DEVICES=0 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-23-1602.23/checkpoint-5"
sleep 10

echo "ckpt 10"
CUDA_VISIBLE_DEVICES=0 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-23-1602.23/checkpoint-10"
sleep 10

echo "ckpt 25"
CUDA_VISIBLE_DEVICES=0 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-23-1602.23/checkpoint-25"
sleep 10

echo "ckpt 100"
CUDA_VISIBLE_DEVICES=0 python3 exp1.py --ckpt_path "/home/u2024001042/workspace/nyx/checkpoint/ft_2025-07-23-1602.23/checkpoint-100"
