import numpy as np
import torch
from shmtensor import DiskTensor
import os


def load_dataset(load_path, pin_memory=False, with_feat=False):
    csc_indptr = DiskTensor(os.path.join(load_path, 'csc_indptr.npy'),
                            pin_memory)
    csc_indices = DiskTensor(os.path.join(load_path, 'csc_indices.npy'),
                             pin_memory)
    train_nids = DiskTensor(os.path.join(load_path, 'train_nids.npy'),
                            pin_memory)
    valid_nids = DiskTensor(os.path.join(load_path, 'valid_nids.npy'),
                            pin_memory)
    test_nids = DiskTensor(os.path.join(load_path, 'test_nids.npy'),
                           pin_memory)
    labels = DiskTensor(os.path.join(load_path, 'labels.npy'), pin_memory)

    if with_feat:
        features = DiskTensor(os.path.join(load_path, 'features.npy'),
                              pin_memory)
    else:
        features = None

    num_classes = int(labels.tensor_[~torch.isnan(labels.tensor_)].max() + 1)
    print(f"num_classes: {num_classes}")

    dataset = {
        'csc_indptr': csc_indptr,
        'csc_indices': csc_indices,
        'train_nids': train_nids,
        'valid_nids': valid_nids,
        'test_nids': test_nids,
        'features': features,
        'labels': labels,
        'num_classes': num_classes
    }

    return dataset


if __name__ == "__main__":
    a = load_dataset('.', pin_memory=True)
    for i in a:
        if i is not None:
            print(i.tensor_.numel(), i.tensor_.shape)
