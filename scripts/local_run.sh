OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 run.py --load-path datasets --model sage --num-trainers 8 --total-epochs 2
