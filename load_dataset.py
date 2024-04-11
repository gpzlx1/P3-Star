import torch
from shmtensor import ShmTensor
import os
import torch.distributed as dist


def open_file(load_path, filename):
    data = torch.load(os.path.join(load_path, filename))
    return data


def load_dataset(meta, load_path):
    tensor_dict = {}
    for key, value in meta.items():
        tensor_dict[key] = open_file(load_path, key + ".pt")
    return tensor_dict


def create_shmtensor(meta, local_rank, world_size, cur_group):
    new_tensor_dict = {}
    for key, value in meta.items():
        new_tensor_dict[key] = ShmTensor(key,
                                         meta[key][1],
                                         local_rank,
                                         world_size,
                                         cur_group,
                                         dtype=meta[key][0])
    return new_tensor_dict


def assign_shmtensor(new_tensor_dict, tensor_dict, local_rank):
    if local_rank == 0:
        for key, value in new_tensor_dict.items():
            new_tensor_dict[key].tensor_[:] = tensor_dict[key]


def dist_load_tensor(load_path):
    meta = torch.load(os.path.join(load_path, 'meta.pt'))
    local_rank, local_world_size, local_group = create_local_group()
    new_tensor_dict = create_shmtensor(meta, local_rank, local_world_size,
                                       local_group)

    if local_rank == 0:
        tensor_dict = load_dataset(meta, load_path)
    dist.barrier(local_group)

    if local_rank == 0:
        assign_shmtensor(new_tensor_dict, tensor_dict, local_rank)
    dist.barrier(local_group)

    labels = new_tensor_dict['labels']
    num_classes = int(labels.tensor_[~torch.isnan(labels.tensor_)].max() + 1)
    new_tensor_dict['num_classes'] = num_classes

    return new_tensor_dict


def create_local_group():
    cur, groups = torch.distributed.new_subgroups()
    return torch.distributed.get_rank(cur), torch.distributed.get_world_size(
        cur), cur


if __name__ == "__main__":
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    tensor_dict = dist_load_tensor('./datasets')
    for key, value in tensor_dict.items():
        print(key, value.tensor_)
