import torch
import torch.nn as nn
import torch.distributed as dist
import tqdm
from dgl import create_block
from models.sage import SageP3Shuffle
import torch.nn.functional as F
import torchmetrics as MF


class P3Trainer:

    def __init__(self, emb_layer, first_layer, other_layer, emb_optimizer,
                 first_optimizer, other_optimizer, loss_fn, hidden_size,
                 nid_type):
        self.emb_layer = emb_layer
        self.first_layer = first_layer
        self.other_layer = other_layer
        self.emb_optimizer = emb_optimizer
        self.first_optimizer = first_optimizer
        self.other_optimizer = other_optimizer
        self.loss_fn = loss_fn
        self.hidden_size = hidden_size

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        est_node_size = 5000
        self.edge_size_lst = [(0, 0, 0, 0)] * self.world_size
        self.input_node_buffer_lst = []
        self.src_edge_buffer_lst = []
        self.dst_edge_buffer_lst = []
        self.global_grad_lst = []
        self.input_feat_buffer_lst = []
        for _ in range(self.world_size):
            self.input_node_buffer_lst.append(
                torch.empty(est_node_size, dtype=nid_type, device='cuda'))
            self.src_edge_buffer_lst.append(
                torch.empty(est_node_size, dtype=nid_type, device='cuda'))
            self.dst_edge_buffer_lst.append(
                torch.empty(est_node_size, dtype=nid_type, device='cuda'))
            self.global_grad_lst.append(
                torch.empty((est_node_size, hidden_size),
                            dtype=torch.float32,
                            device='cuda'))
            self.input_feat_buffer_lst.append(
                torch.empty((est_node_size, hidden_size),
                            dtype=torch.float32,
                            device='cuda'))
        self.local_hid_buffer_lst: list[torch.Tensor] = [None
                                                         ] * self.world_size

    def train_one_epoch(self, dataloader, labels):
        self.emb_layer.train()
        self.first_layer.train()
        self.other_layer.train()
        for i, (input_nodes, output_nodes, blocks) in enumerate(
                tqdm.tqdm(dataloader, disable=self.rank != 0)):
            top_block = blocks[0]

            src, dst = top_block.adj_tensors('coo')
            self.edge_size_lst[self.rank] = (self.rank, src.shape[0],
                                             top_block.num_src_nodes(),
                                             top_block.num_dst_nodes())

            dist.all_gather_object(object_list=self.edge_size_lst,
                                   obj=self.edge_size_lst[self.rank])

            for r, edge_size, src_node_size, dst_node_size in self.edge_size_lst:
                self.src_edge_buffer_lst[r].resize_(edge_size)
                self.dst_edge_buffer_lst[r].resize_(edge_size)
                self.input_node_buffer_lst[r].resize_(src_node_size)

            handle1 = dist.all_gather(tensor_list=self.input_node_buffer_lst,
                                      tensor=input_nodes,
                                      async_op=True)
            handle2 = dist.all_gather(tensor_list=self.src_edge_buffer_lst,
                                      tensor=src,
                                      async_op=True)
            handle3 = dist.all_gather(tensor_list=self.dst_edge_buffer_lst,
                                      tensor=dst,
                                      async_op=True)
            handle1.wait()

            for r, _input_nodes in enumerate(self.input_node_buffer_lst):
                self.input_feat_buffer_lst[r] = self.emb_layer(
                    _input_nodes) # .cuda()

            handle2.wait()
            handle3.wait()

            block = None
            for r in range(self.world_size):
                # input_nodes = self.input_node_buffer_lst[r]
                input_feats = self.input_feat_buffer_lst[r]
                if r == self.rank:
                    block = top_block
                else:
                    src = self.src_edge_buffer_lst[r]
                    dst = self.dst_edge_buffer_lst[r]
                    src_node_size = self.edge_size_lst[r][2]
                    dst_node_size = self.edge_size_lst[r][3]
                    block = create_block(('coo', (src, dst)),
                                         num_dst_nodes=dst_node_size,
                                         num_src_nodes=src_node_size,
                                         device='cuda')

                self.local_hid_buffer_lst[r] = self.first_layer(
                    block, input_feats)
                self.global_grad_lst[r].resize_(
                    [block.num_dst_nodes(), self.hidden_size])

            local_hid = SageP3Shuffle.apply(
                self.rank, self.world_size,
                self.local_hid_buffer_lst[self.rank],
                self.local_hid_buffer_lst, self.global_grad_lst)
            output_labels = labels[output_nodes.cpu()].cuda()
            output_pred = self.other_layer(blocks[1:], local_hid)
            loss = F.cross_entropy(output_pred, output_labels)

            self.emb_optimizer.zero_grad()
            self.first_optimizer.zero_grad()
            self.other_optimizer.zero_grad()

            loss.backward()

            self.other_optimizer.step()

            for r, global_grad in enumerate(self.global_grad_lst):
                if r != self.rank:
                    self.local_hid_buffer_lst[r].backward(global_grad)

            self.first_optimizer.step()
            self.emb_optimizer.step()
            
            

    def inference(self, dataloader, labels, num_classes):
        self.emb_layer.eval()
        self.first_layer.eval()
        self.other_layer.eval()

        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                tqdm.tqdm(dataloader, disable=self.rank != 0)):
            with torch.no_grad():
                top_block = blocks[0]
                # 1. Send and Receive edges for all the other gpus
                src, dst = top_block.adj_tensors('coo')  # dgl v1.1 and above
                # src, dst = top_block.adj_sparse(fmt='coo') # dgl v1.0 and below
                self.edge_size_lst[self.rank] = (
                    self.rank, src.shape[0], top_block.num_src_nodes(),
                    top_block.num_dst_nodes()
                )  # rank, edge_size, input_node_size
                dist.all_gather_object(object_list=self.edge_size_lst,
                                       obj=self.edge_size_lst[self.rank])
                for r, edge_size, src_node_size, dst_node_size in self.edge_size_lst:
                    self.src_edge_buffer_lst[r].resize_(edge_size)
                    self.dst_edge_buffer_lst[r].resize_(edge_size)
                    self.input_node_buffer_lst[r].resize_(src_node_size)
                    # input_feat_buffer_lst[rank].resize_([src_node_size, local_feat_width])
                # dist.barrier()
                handle1 = dist.all_gather(
                    tensor_list=self.input_node_buffer_lst,
                    tensor=input_nodes,
                    async_op=True)
                handle2 = dist.all_gather(tensor_list=self.src_edge_buffer_lst,
                                          tensor=src,
                                          async_op=True)
                handle3 = dist.all_gather(tensor_list=self.dst_edge_buffer_lst,
                                          tensor=dst,
                                          async_op=True)
                handle1.wait()
                for r, _input_nodes in enumerate(self.input_node_buffer_lst):
                    self.input_feat_buffer_lst[r] = self.emb_layer(
                        _input_nodes).cuda()
                handle2.wait()
                handle3.wait()
                # 3. Fetch feature data and compute hid feature for other GPUs
                block = None
                for r in range(self.world_size):
                    input_nodes = self.input_node_buffer_lst[r]
                    input_feats = self.input_feat_buffer_lst[r]
                    if r == self.rank:
                        block = top_block
                    else:
                        src = self.src_edge_buffer_lst[r]
                        dst = self.dst_edge_buffer_lst[r]
                        src_node_size = self.edge_size_lst[r][2]
                        dst_node_size = self.edge_size_lst[r][3]
                        block = create_block(('coo', (src, dst)),
                                             num_dst_nodes=dst_node_size,
                                             num_src_nodes=src_node_size,
                                             device='cuda')

                    self.local_hid_buffer_lst[r] = self.first_layer(
                        block, input_feats)
                    # global_grad_lst[r].resize_([block.num_dst_nodes(), hid_feats])
                    del block

                local_hid = SageP3Shuffle.apply(
                    self.rank, self.world_size,
                    self.local_hid_buffer_lst[self.rank],
                    self.local_hid_buffer_lst, None)
                ys.append(labels[output_nodes.cpu()].cpu())
                y_hats.append(self.other_layer(blocks[1:], local_hid).cpu())

        acc = MF.Accuracy(task="multiclass", num_classes=num_classes)
        acc = acc(torch.cat(y_hats), torch.cat(ys)).cuda()

        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        return (acc / self.world_size).item()
