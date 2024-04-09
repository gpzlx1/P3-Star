import dgl
import torch
import os
import argparse
import dgl
import ogb
import numpy as np


def convert(dgl_graph: dgl.graph, save_feat: bool, path: str):
    csc = dgl_graph.adj_tensors('csc')
    indptr, indices, _ = csc
    train_nids = dgl_graph.ndata['train_mask']
    valid_nids = dgl_graph.ndata['test']
    test_nids = dgl_graph.ndata['test']
    labels = dgl_graph.ndata['labels']

    if save_feat:
        features = dgl_graph.ndata['features']

    np.save(os.path.join(path, 'csc_indptr'), indptr.numpy())
    np.save(os.path.join(path, 'csc_indices'), indices.numpy())
    np.save(os.path.join(path, 'train_nids'), train_nids.numpy())
    np.save(os.path.join(path, 'valid_nids'), valid_nids.numpy())
    np.save(os.path.join(path, 'test_nids'), test_nids.numpy())
    np.save(os.path.join(path, 'labels'), test_nids.numpy())

    if save_feat:
        np.save(os.path.join(path, 'features'), features.numpy())


def process_reddit(save_path, save_feat=False):
    data = dgl.data.RedditDataset(self_loop=True)
    dgl_graph = data[0]

    csc = dgl_graph.adj_tensors('csc')
    indptr, indices, _ = csc
    train_nids = torch.nonzero(dgl_graph.ndata['train_mask']).squeeze(1)
    valid_nids = torch.nonzero(dgl_graph.ndata['val_mask']).squeeze(1)
    test_nids = torch.nonzero(dgl_graph.ndata['test_mask']).squeeze(1)
    labels = dgl_graph.ndata['label']

    if save_feat:
        features = dgl_graph.ndata['feat']

    np.save(os.path.join(save_path, 'csc_indptr'), indptr.numpy())
    np.save(os.path.join(save_path, 'csc_indices'), indices.numpy())
    np.save(os.path.join(save_path, 'train_nids'), train_nids.numpy())
    np.save(os.path.join(save_path, 'valid_nids'), valid_nids.numpy())
    np.save(os.path.join(save_path, 'test_nids'), test_nids.numpy())
    np.save(os.path.join(save_path, 'labels'), labels.numpy())

    if save_feat:
        np.save(os.path.join(save_path, 'features'), features.numpy())


if __name__ == '__main__':
    process_reddit('./datasets', save_feat=True)
