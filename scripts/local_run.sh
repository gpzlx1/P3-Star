OMP_NUM_THREADS=8 torchrun --nproc-per-node 2 run.py --load-path datasets --model sage --num-trainers 2 --total-epochs 2
