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
    local_embedding_size = args.embedding_size // nccl_world_size
    embedding_feature = Embedding(dataset['csc_indptr'].tensor_.numel() - 1,
                                  local_embedding_size)

    #local_embedding_size = dataset['features'].tensor_.size(1)
    #embedding_feature = dataset['features'].tensor_

    labels = dataset['labels'].tensor_
    train_nids = dataset['train_nids'].tensor_
    test_nids = dataset['test_nids'].tensor_
    g = dgl.graph(
        ('csr', (dataset['csc_indptr'].tensor_, dataset['csc_indices'].tensor_,
                 torch.Tensor())))

    # set model
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
            global_model, device_ids=[nccl_rank])

    # set optimizer
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
    emb_optimizer = SparseAdam((embedding_feature, ), lr=args.sparse_lr)

    # train
    # set dataloader
    sampler = dgl.dataloading.NeighborSampler(args.fanouts)
    dataloader = dgl.dataloading.DataLoader(g,
                                            train_nids,
                                            sampler,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=False,
                                            use_ddp=True,
                                            use_uva=True)

    trainer = P3Trainer(embedding_feature, local_model, global_model,
                        emb_optimizer, local_optimizer, global_optimizer,
                        F.cross_entropy, hidden_size,
                        dataset['csc_indptr'].tensor_.dtype)

    # train
    for i in range(10):
        trainer.train_one_epoch(dataloader, labels)

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
        acc = trainer.inference(test_dataloader, labels,
                                dataset['num_classes'])
        if nccl_rank == 0:
            print(acc)


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
    args = parser.parse_args()
    print(args)
    args.fanouts = [int(i) for i in args.fanouts.split(',')]
    args.heads = [int(i) for i in args.heads.split(',')]

    if args.model == 'gat':
        args.hidden_size = args.hidden_size // args.heads[0]

    # init group
    dist.init_process_group('nccl', init_method='env://')

    ## set device
    torch.cuda.set_device(dist.get_rank())

    print("load dataset")
    dataset = load_dataset(args.load_path, pin_memory=False, with_feat=True)
    print("finish load dataset")

    main(args, dataset)
