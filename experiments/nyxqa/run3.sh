echo "dim 1024"
CUDA_VISIBLE_DEVICES=7 python3 exp2.py --output_dim 1024
sleep 10

echo "dim 512"
CUDA_VISIBLE_DEVICES=7 python3 exp2.py --output_dim 512
sleep 10

echo "dim 256"
CUDA_VISIBLE_DEVICES=7 python3 exp2.py --output_dim 256
sleep 10

echo "dim 128"
CUDA_VISIBLE_DEVICES=7 python3 exp2.py --output_dim 128