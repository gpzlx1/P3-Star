# Contruct a two-layer GNN model
from dgl.nn.pytorch.conv import GATConv
import torch.nn as nn
import torch
import torch.distributed as dist


class Gat(nn.Module):

    def __init__(self,
                 in_feats: int,
                 hid_feats: int,
                 num_layers: int,
                 out_feats: int,
                 num_heads=[8, 8, 1],
                 activation=nn.functional.relu,
                 feat_dropout=0.1,
                 attn_dropout=0.1):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = in_feats if layer_idx == 0 else hid_feats * num_heads[
                layer_idx - 1]
            out_dim = out_feats if layer_idx == num_layers - 1 else hid_feats
            layer_activation = None if layer_idx == num_layers - 1 else activation
            self.layers.append(
                GATConv(in_dim,
                        out_dim,
                        num_heads[layer_idx],
                        feat_drop=feat_dropout,
                        attn_drop=attn_dropout,
                        activation=layer_activation,
                        allow_zero_in_degree=True))
        self.fwd_l1_timer = []
        self.hid_feats_lst = []

    def forward(self, blocks, feat):
        hid_feats = feat
        l1_start = torch.cuda.Event(enable_timing=True)
        l1_start.record()
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if (layer_idx == 0):
                l1_end = torch.cuda.Event(enable_timing=True)
                l1_end.record()
                self.fwd_l1_timer.append((l1_start, l1_end))
            if layer_idx != len(self.layers) - 1:
                hid_feats = hid_feats.mean(1)
            else:
                hid_feats = hid_feats.flatten(1)
        return hid_feats

    def fwd_l1_time(self):
        torch.cuda.synchronize()
        fwd_time = 0.0
        for l1_start, l1_end in self.fwd_l1_timer:
            fwd_time += l1_start.elapsed_time(l1_end)
        self.fwd_l1_timer = []
        return fwd_time


class GatP3Shuffle(torch.autograd.Function):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(ctx, self_rank: int, world_size: int, local_hid: torch.Tensor,
                local_hids: list[torch.Tensor],
                global_grads: list[torch.Tensor]) -> torch.Tensor:
        # print(f"forward {self_rank=} {world_size=} {local_hid.shape}")
        ctx.self_rank = self_rank
        ctx.world_size = world_size
        ctx.global_grads = global_grads
        # aggregated_hid = torch.clone(local_hid)
        aggregated_hid = local_hid.detach().clone()
        handle = None
        for r in range(world_size):
            if r == self_rank:
                handle = dist.reduce(
                    tensor=aggregated_hid, dst=r,
                    async_op=True)  # gathering data from other GPUs
            else:
                dist.reduce(tensor=local_hids[r], dst=r, async_op=True
                            )  # TODO: Async gathering data from other GPUs
        handle.wait()
        return aggregated_hid

    @staticmethod
    def backward(ctx, grad_outputs):
        # print(f"self.rank={ctx.self_rank} send_grad_shape={grad_outputs.shape} global_grads_shape={[x.shape for x in ctx.global_grads]}")
        dist.all_gather(tensor=grad_outputs, tensor_list=ctx.global_grads)
        return None, None, grad_outputs, None, None


class GatP3First(nn.Module):

    def __init__(self, in_feats: int, hid_feats: int, num_heads: int):
        super().__init__()
        self.conv = GATConv(in_feats=in_feats,
                            out_feats=int(hid_feats / num_heads),
                            num_heads=num_heads)

    def forward(self, block, feat):
        return self.conv(block, feat).flatten(1)


class GatP3(nn.Module):

    def __init__(self,
                 in_feats: int,
                 hid_feats: int,
                 num_layers: int,
                 out_feats: int,
                 num_heads=[8, 8, 1],
                 activation=nn.functional.relu,
                 feat_dropout=0.1,
                 attn_dropout=0.1):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = in_feats if layer_idx == 0 else hid_feats * num_heads[
                layer_idx - 1]
            out_dim = out_feats if layer_idx == num_layers - 1 else hid_feats
            layer_activation = None if layer_idx == num_layers - 1 else activation

            if layer_idx == 0:
                continue
            else:
                self.layers.append(
                    GATConv(in_dim,
                            out_dim,
                            num_heads[layer_idx],
                            feat_drop=feat_dropout,
                            attn_drop=attn_dropout,
                            activation=layer_activation,
                            allow_zero_in_degree=True))
        self.hid_feats_lst = []

    def forward(self, blocks, feat):
        hid_feats = feat
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if layer_idx != len(self.layers) - 1:
                hid_feats = hid_feats.mean(1)
            else:
                hid_feats = hid_feats.flatten(1)
        return hid_feats


def create_gat_p3(in_feats: int,
                  hid_feats: int,
                  num_classes: int,
                  num_layers: int,
                  num_heads: int = 4) -> tuple[nn.Module, nn.Module]:
    first_layer = GatP3First(in_feats, hid_feats,
                             num_heads)  # Intra-Model Parallel
    remain_layers = GatP3(in_feats,
                          hid_feats,
                          num_layers,
                          num_classes,
                          num_heads=num_heads)  # Data Parallel
    return (first_layer, remain_layers)
