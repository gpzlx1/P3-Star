import os
import torch
import dgl
from load_dataset import load_dataset
import torch.distributed as dist
import argparse
from models.sage import create_sage_p3
from models.sage import SageP3Shuffle
from models.gat import create_gat_p3
import time
from dgl import create_block
import tqdm
import numpy as np
import torch.nn.functional as F
import torchmetrics as MF
from embedding import Embedding, SparseAdam
from trainer import P3Trainer


def main(args, dataset):
    # set rank and world size
    nccl_rank = dist.get_rank()
    print(nccl_rank)
    nccl_world_size = dist.get_world_size()

    # local embedding feature
    if nccl_rank == 0:
        print("Create p3 embedding")
    local_embedding_size = args.embedding_size // nccl_world_size
    embedding_feature = Embedding(dataset['csc_indptr'].tensor_.numel() - 1,
                                  local_embedding_size)

    #local_embedding_size = dataset['features'].tensor_.size(1)
    #embedding_feature = dataset['features'].tensor_

    if nccl_rank == 0:
        print("Create dgl graph")
    labels = dataset['labels'].tensor_
    train_nids = dataset['train_nids'].tensor_
    test_nids = dataset['test_nids'].tensor_
    valid_nids = dataset['valid_nids'].tensor_
    g = dgl.graph(
        ('csr', (dataset['csc_indptr'].tensor_, dataset['csc_indices'].tensor_,
                 torch.Tensor())))

    # set model
    if nccl_rank == 0:
        print("Create p3 model")
    if args.model == "sage":
        hidden_size = args.hidden_size
        local_model, global_model = create_sage_p3(local_embedding_size,
                                                   args.hidden_size,
                                                   dataset['num_classes'],
                                                   args.num_layers)
    elif args.model == "gat":
        hidden_size = args.hidden_size * args.heads[0]
        local_model, global_model = create_gat_p3(local_embedding_size,
                                                  args.hidden_size,
                                                  dataset['num_classes'],
                                                  args.num_layers,
                                                  heads=args.heads)

    local_model = local_model.cuda()
    global_model = global_model.cuda()

    if nccl_world_size > 1:
        global_model = torch.nn.parallel.DistributedDataParallel(
            global_model, device_ids=[nccl_rank % args.num_trainers])

    # set optimizer
    if nccl_rank == 0:
        print("Create optimizers")
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
    emb_optimizer = SparseAdam((embedding_feature, ), lr=args.sparse_lr)

    # train
    # set dataloader
    if nccl_rank == 0:
        print("Create dgl sampler and dataloader")
    sampler = dgl.dataloading.NeighborSampler(args.fanouts)
    dataloader = dgl.dataloading.DataLoader(g,
                                            train_nids,
                                            sampler,
                                            device="cuda",
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=False,
                                            use_ddp=True,
                                            use_uva=True)

    if nccl_rank == 0:
        print("Create p3 trainer")
    trainer = P3Trainer(embedding_feature, local_model, global_model,
                        emb_optimizer, local_optimizer, global_optimizer,
                        F.cross_entropy, hidden_size,
                        dataset['csc_indptr'].tensor_.dtype)

    # train
    if nccl_rank == 0:
        print("Start training")
    timelst = []
    for i in range(args.total_epochs):
        tic = time.time()
        trainer.train_one_epoch(dataloader, labels)
        toc = time.time()
        epoch_time = toc - tic
        reduce_tensor = torch.tensor([epoch_time]).cuda()
        dist.all_reduce(reduce_tensor, dist.ReduceOp.SUM)
        epoch_time = reduce_tensor[0].item() / dist.get_world_size()
        timelst.append(epoch_time)

        # inference
        test_dataloader = dgl.dataloading.DataLoader(
            g,
            test_nids,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            use_ddp=True,
            use_uva=True)
        test_acc = trainer.inference(test_dataloader, labels,
                                     dataset['num_classes'])
        valid_dataloader = dgl.dataloading.DataLoader(
            g,
            valid_nids,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            use_ddp=True,
            use_uva=True)
        valid_acc = trainer.inference(valid_dataloader, labels,
                                      dataset['num_classes'])
        if nccl_rank == 0:
            print("Epoch time: {:.3f} s, Valid acc: {:.3f}, Test acc: {:.3f}".
                  format(epoch_time, valid_acc, test_acc))
    if nccl_rank == 0:
        print("Avg epoch time: {:.3f} s".format(np.mean(timelst[1:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-path', type=str, required=True)
    parser.add_argument('--graph-name', type=str)
    parser.add_argument('--total-epochs',
                        default=6,
                        type=int,
                        help='Total epochs to train the model')
    parser.add_argument('--hidden-size',
                        default=256,
                        type=int,
                        help='Size of a hidden feature')
    parser.add_argument('--embedding-size',
                        default=256,
                        type=int,
                        help='Size of a hidden feature')
    parser.add_argument('--batch-size',
                        default=1000,
                        type=int,
                        help='Input batch size on each device (default: 1024)')
    parser.add_argument('--model',
                        default="gat",
                        type=str,
                        help='Model type: sage or gat',
                        choices=['sage', 'gat'])
    parser.add_argument("--heads", type=str, default="8,8,1")
    parser.add_argument('--num-layers', default=3, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--sparse-lr', default=1e-2, type=float)
    parser.add_argument('--fanouts', default="5, 10, 15", type=str)
    parser.add_argument('--num-trainers', default=2, type=int)
    args = parser.parse_args()
    print(args)
    args.fanouts = [int(i) for i in args.fanouts.split(',')]
    args.heads = [int(i) for i in args.heads.split(',')]

    if args.model == 'gat':
        args.hidden_size = args.hidden_size // args.heads[0]

    # init group
    dist.init_process_group('nccl', init_method='env://')

    ## set device
    torch.cuda.set_device(dist.get_rank() % args.num_trainers)

    print("load dataset")
    dataset = load_dataset(args.load_path, pin_memory=False, with_feat=False)
    print("finish load dataset")

    main(args, dataset)
