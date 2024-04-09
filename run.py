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

    # set model
    if args.model == "sage":
        local_model, global_model = create_sage_p3(local_embedding_size,
                                                   args.hidden_size,
                                                   dataset['num_classes'],
                                                   args.num_layers)
    elif args.model == "gat":
        local_model, global_model = create_gat_p3(local_embedding_size,
                                                  args.hidden_size,
                                                  dataset['num_classes'],
                                                  args.num_layers,
                                                  num_heads=args.num_heads)

    local_model = local_model.cuda()
    global_model = global_model.cuda()

    if nccl_world_size > 1:
        global_model = torch.nn.parallel.DistributedDataParallel(
            global_model, device_ids=[nccl_rank])

    # set optimizer
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
    emb_optimizer = SparseAdam((embedding_feature, ), lr=args.sparse_lr)

    # begin training
    est_node_size = args.batch_size * 20
    edge_size_lst = [(0, 0, 0, 0)] * nccl_world_size
    input_node_buffer_lst = []
    src_edge_buffer_lst = []
    dst_edge_buffer_lst = []
    global_grad_lst = []
    input_feat_buffer_lst = []
    for _ in range(nccl_world_size):
        input_node_buffer_lst.append(
            torch.empty(est_node_size,
                        dtype=dataset['csc_indptr'].tensor_.dtype,
                        device='cuda'))
        src_edge_buffer_lst.append(
            torch.empty(est_node_size,
                        dtype=dataset['csc_indptr'].tensor_.dtype,
                        device='cuda'))
        dst_edge_buffer_lst.append(
            torch.empty(est_node_size,
                        dtype=dataset['csc_indptr'].tensor_.dtype,
                        device='cuda'))
        global_grad_lst.append(
            torch.empty((est_node_size, args.hidden_size),
                        dtype=torch.float32,
                        device='cuda'))
        input_feat_buffer_lst.append(
            torch.empty((est_node_size, args.hidden_size),
                        dtype=torch.float32,
                        device='cuda'))
    local_hid_buffer_lst: list[torch.Tensor] = [None] * nccl_world_size

    print('begin training')

    # for one epoch
    begin = time.time()
    for i, (input_nodes, output_nodes,
            blocks) in enumerate(tqdm.tqdm(dataloader, disable=nccl_rank
                                           != 0)):
        top_block = blocks[0]

        src, dst = top_block.adj_tensors('coo')
        edge_size_lst[nccl_rank] = (nccl_rank, src.shape[0],
                                    top_block.num_src_nodes(),
                                    top_block.num_dst_nodes())

        dist.all_gather_object(object_list=edge_size_lst,
                               obj=edge_size_lst[nccl_rank])

        for rank, edge_size, src_node_size, dst_node_size in edge_size_lst:
            src_edge_buffer_lst[rank].resize_(edge_size)
            dst_edge_buffer_lst[rank].resize_(edge_size)
            input_node_buffer_lst[rank].resize_(src_node_size)

        handle1 = dist.all_gather(tensor_list=input_node_buffer_lst,
                                  tensor=input_nodes,
                                  async_op=True)
        handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst,
                                  tensor=src,
                                  async_op=True)
        handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst,
                                  tensor=dst,
                                  async_op=True)
        handle1.wait()

        for rank, _input_nodes in enumerate(input_node_buffer_lst):
            input_feat_buffer_lst[rank] = embedding_feature(
                _input_nodes.cpu()).cuda()

        handle2.wait()
        handle3.wait()

        block = None
        for r in range(nccl_world_size):
            input_nodes = input_node_buffer_lst[r]
            input_feats = input_feat_buffer_lst[r]
            if r == nccl_rank:
                block = top_block
            else:
                src = src_edge_buffer_lst[r]
                dst = dst_edge_buffer_lst[r]
                src_node_size = edge_size_lst[r][2]
                dst_node_size = edge_size_lst[r][3]
                block = create_block(('coo', (src, dst)),
                                     num_dst_nodes=dst_node_size,
                                     num_src_nodes=src_node_size,
                                     device='cuda')

            local_hid_buffer_lst[r] = local_model(block, input_feats)
            global_grad_lst[r].resize_(
                [block.num_dst_nodes(), args.hidden_size])

        local_hid = SageP3Shuffle.apply(nccl_rank, nccl_world_size,
                                        local_hid_buffer_lst[nccl_rank],
                                        local_hid_buffer_lst, global_grad_lst)
        output_labels = labels[output_nodes.cpu()].cuda()
        output_pred = global_model(blocks[1:], local_hid)
        loss = F.cross_entropy(output_pred, output_labels)

        global_optimizer.zero_grad()
        local_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()

        # embedding

        for r, global_grad in enumerate(global_grad_lst):
            if r != nccl_rank:
                local_optimizer.zero_grad()
                local_hid_buffer_lst[r].backward(global_grad)

        local_optimizer.step()
        emb_optimizer.zero_grad()
        emb_optimizer.step()

    # end = time.time()

    # print(end - begin)

    test_dataloader = dgl.dataloading.DataLoader(g,
                                                 test_nids,
                                                 sampler,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 drop_last=False,
                                                 use_ddp=True,
                                                 use_uva=True)

    # inference
    if True:
        global_model.eval()
        local_model.eval()

        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                tqdm.tqdm(test_dataloader, disable=nccl_rank != 0)):
            with torch.no_grad():
                top_block = blocks[0]
                # 1. Send and Receive edges for all the other gpus
                src, dst = top_block.adj_tensors('coo')  # dgl v1.1 and above
                # src, dst = top_block.adj_sparse(fmt='coo') # dgl v1.0 and below
                edge_size_lst[nccl_rank] = (
                    nccl_rank, src.shape[0], top_block.num_src_nodes(),
                    top_block.num_dst_nodes()
                )  # rank, edge_size, input_node_size
                dist.all_gather_object(object_list=edge_size_lst,
                                       obj=edge_size_lst[nccl_rank])
                for rank, edge_size, src_node_size, dst_node_size in edge_size_lst:
                    src_edge_buffer_lst[rank].resize_(edge_size)
                    dst_edge_buffer_lst[rank].resize_(edge_size)
                    input_node_buffer_lst[rank].resize_(src_node_size)
                    # input_feat_buffer_lst[rank].resize_([src_node_size, local_feat_width])
                # dist.barrier()
                handle1 = dist.all_gather(tensor_list=input_node_buffer_lst,
                                          tensor=input_nodes,
                                          async_op=True)
                handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst,
                                          tensor=src,
                                          async_op=True)
                handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst,
                                          tensor=dst,
                                          async_op=True)
                handle1.wait()
                for rank, _input_nodes in enumerate(input_node_buffer_lst):
                    input_feat_buffer_lst[rank] = embedding_feature(
                        _input_nodes.to('cpu')).cuda()
                handle2.wait()
                handle3.wait()
                # 3. Fetch feature data and compute hid feature for other GPUs
                block = None
                for r in range(nccl_world_size):
                    input_nodes = input_node_buffer_lst[r]
                    input_feats = input_feat_buffer_lst[r]
                    if r == nccl_rank:
                        block = top_block
                    else:
                        src = src_edge_buffer_lst[r]
                        dst = dst_edge_buffer_lst[r]
                        src_node_size = edge_size_lst[r][2]
                        dst_node_size = edge_size_lst[r][3]
                        block = create_block(('coo', (src, dst)),
                                             num_dst_nodes=dst_node_size,
                                             num_src_nodes=src_node_size,
                                             device='cuda')

                    local_hid_buffer_lst[r] = local_model(block, input_feats)
                    # global_grad_lst[r].resize_([block.num_dst_nodes(), hid_feats])
                    del block

                local_hid = SageP3Shuffle.apply(
                    nccl_rank, nccl_world_size,
                    local_hid_buffer_lst[nccl_rank], local_hid_buffer_lst,
                    None)
                ys.append(labels[output_nodes.cpu()].cpu())
                y_hats.append(global_model(blocks[1:], local_hid).cpu())

        acc = MF.Accuracy(task="multiclass",
                          num_classes=dataset['num_classes'])
        acc = acc(torch.cat(y_hats), torch.cat(ys)).cuda()

        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        if nccl_rank == 0:
            print((acc / nccl_world_size).item())


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
                        default=1024,
                        type=int,
                        help='Input batch size on each device (default: 1024)')
    parser.add_argument('--model',
                        default="gat",
                        type=str,
                        help='Model type: sage or gat',
                        choices=['sage', 'gat'])
    parser.add_argument('--num_heads',
                        default=4,
                        type=int,
                        help='Number of heads for GAT model')
    parser.add_argument('--num-layers', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--sparse-lr', default=0.01, type=float)
    parser.add_argument('--fanouts', default="10, 10, 10", type=str)
    parser.add_argument('--dropout', default=0.5, type=float)
    args = parser.parse_args()
    print(args)
    args.fanouts = [int(i) for i in args.fanouts.split(',')]

    # init group
    dist.init_process_group('nccl', init_method='env://')

    ## set device
    torch.cuda.set_device(dist.get_rank())

    print("load dataset")
    dataset = load_dataset(args.load_path, pin_memory=False, with_feat=True)
    print("finish load dataset")

    main(args, dataset)
