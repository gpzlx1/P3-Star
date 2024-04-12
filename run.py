import torch
import dgl
from load_dataset import dist_load_tensor
import torch.distributed as dist
import argparse
from models.sage import create_sage_p3
from models.gat import create_gat_p3
import time
import numpy as np
import torch.nn.functional as F
from embedding import Embedding, SparseAdam, SparseAdagrad
from trainer import P3Trainer
from shmtensor import GPUSamplingDataloader


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
    if args.emb_optim == "adam":
        emb_optimizer = SparseAdam((embedding_feature, ), lr=args.sparse_lr)
    elif args.emb_optim == "adagrad":
        emb_optimizer = SparseAdagrad((embedding_feature, ), lr=args.sparse_lr)

    # set dataloader
    if nccl_rank == 0:
        print("Create gpu sampling dataloader")
    dataloader = GPUSamplingDataloader(dataset['csc_indptr'].tensor_,
                                       dataset['csc_indices'].tensor_,
                                       train_nids,
                                       args.batch_size,
                                       args.fanouts,
                                       shuffle=True,
                                       drop_last=False,
                                       use_ddp=True)
    if nccl_rank == 0:
        print("Start presampling")
    tic = time.time()
    sampling_hotness, embedding_hotness = dataloader.presampling()
    toc = time.time()
    if nccl_rank == 0:
        print("Presampling time: {:.3f} s".format(toc - tic))

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
        if i == 1:
            # create GPU cache
            if nccl_rank == 0:
                print("Create GPU cache")
            capacity = torch.cuda.mem_get_info(torch.cuda.current_device(
            ))[1] - torch.cuda.max_memory_allocated(
            ) - args.reversed_cuda_mem * 1024 * 1024 * 1024
            embedding_item_size = embedding_feature.tensor.element_size(
            ) + emb_optimizer.itemsize(embedding_feature.name)
            avg_degree = dataset['csc_indices'].tensor_.numel() / (
                dataset['csc_indptr'].tensor_.numel() - 1)
            sampling_item_size = avg_degree * dataset[
                'csc_indices'].tensor_.element_size(
                ) + dataset['csc_indptr'].tensor_.element_size()
            embedding_capacity = int(
                capacity * embedding_item_size /
                (embedding_item_size + sampling_item_size))
            sampling_capacity = int(capacity * sampling_item_size /
                                    (embedding_item_size + sampling_item_size))
            embedding_feature.create_cache(embedding_capacity,
                                           embedding_hotness, emb_optimizer)
            dataloader.create_cache(sampling_capacity, sampling_hotness)

        tic = time.time()
        dataloader.update_params(seeds=train_nids)
        trainer.train_one_epoch(dataloader, labels)
        toc = time.time()
        epoch_time = toc - tic
        reduce_tensor = torch.tensor([epoch_time]).cuda()
        dist.all_reduce(reduce_tensor, dist.ReduceOp.SUM)
        epoch_time = reduce_tensor[0].item() / dist.get_world_size()
        timelst.append(epoch_time)

        # inference
        dataloader.update_params(seeds=test_nids)
        test_acc = trainer.inference(dataloader, labels,
                                     dataset['num_classes'])

        dataloader.update_params(seeds=valid_nids)
        valid_acc = trainer.inference(dataloader, labels,
                                      dataset['num_classes'])

        # reduce valid_acc, test_acc
        valid_acc = torch.Tensor([valid_acc]).cuda()
        dist.all_reduce(valid_acc, dist.ReduceOp.SUM)
        test_acc = torch.Tensor([test_acc]).cuda()
        dist.all_reduce(test_acc, dist.ReduceOp.SUM)
        if nccl_rank == 0:
            print("Epoch time: {:.3f} s, Valid acc: {:.3f}, Test acc: {:.3f}".
                  format(epoch_time,
                         valid_acc.item() / dist.get_world_size(),
                         test_acc.item() / dist.get_world_size()))

    # reduce total epoch time
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
    parser.add_argument('--emb-optim',
                        default="adam",
                        type=str,
                        help='Embedding optimizer type: adam or adagrad',
                        choices=['adam', 'adagrad'])
    parser.add_argument('--sparse-lr', default=1e-2, type=float)
    parser.add_argument('--fanouts', default="5, 10, 15", type=str)
    parser.add_argument('--num-trainers', default=2, type=int)
    parser.add_argument('--reversed-cuda-mem', default=2.0, type=float)
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

    ## create local group

    print("load dataset")
    dataset = dist_load_tensor(args.load_path)
    print("finish load dataset")

    main(args, dataset)
