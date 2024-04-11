import dgl
import torch
import os


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

    torch.save(indptr, os.path.join(save_path, 'csc_indptr.pt'))
    torch.save(indices, os.path.join(save_path, 'csc_indices.pt'))
    torch.save(train_nids, os.path.join(save_path, 'train_nids.pt'))
    torch.save(valid_nids, os.path.join(save_path, 'valid_nids.pt'))
    torch.save(test_nids, os.path.join(save_path, 'test_nids.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))

    if save_feat:
        torch.save(features, os.path.join(save_path, 'features.pt'))

    meta = {
        "csc_indptr": (indptr.dtype, indptr.shape),
        "csc_indices": (indices.dtype, indices.shape),
        "train_nids": (train_nids.dtype, train_nids.shape),
        "valid_nids": (valid_nids.dtype, valid_nids.shape),
        "test_nids": (test_nids.dtype, test_nids.shape),
        "labels": (labels.dtype, labels.shape)
    }

    if save_feat:
        meta["features"] = (features.dtype, features.shape)

    torch.save(meta, os.path.join(save_path, 'meta.pt'))


def process_ogbn(name, root, save_path, save_feat=False):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(name=name, root=root)
    dgl_graph, labels = data[0]
    splitted_idx = data.get_idx_split()

    csc = dgl_graph.adj_tensors('csc')
    indptr, indices, _ = csc
    train_nids = splitted_idx["train"]
    valid_nids = splitted_idx["valid"]
    test_nids = splitted_idx["test"]
    labels = labels[:, 0]

    if save_feat:
        features = dgl_graph.ndata['feat']

    torch.save(indptr, os.path.join(save_path, 'csc_indptr.pt'))
    torch.save(indices, os.path.join(save_path, 'csc_indices.pt'))
    torch.save(train_nids, os.path.join(save_path, 'train_nids.pt'))
    torch.save(valid_nids, os.path.join(save_path, 'valid_nids.pt'))
    torch.save(test_nids, os.path.join(save_path, 'test_nids.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))

    if save_feat:
        torch.save(features, os.path.join(save_path, 'features.pt'))

    meta = {
        "csc_indptr": (indptr.dtype, indptr.shape),
        "csc_indices": (indices.dtype, indices.shape),
        "train_nids": (train_nids.dtype, train_nids.shape),
        "valid_nids": (valid_nids.dtype, valid_nids.shape),
        "test_nids": (test_nids.dtype, test_nids.shape),
        "labels": (labels.dtype, labels.shape)
    }

    if save_feat:
        meta["features"] = (features.dtype, features.shape)

    torch.save(meta, os.path.join(save_path, 'meta.pt'))


if __name__ == '__main__':
    process_reddit('./datasets', save_feat=False)
    #process_ogbn("ogbn-papers100M",
    #             "/home/ubuntu/workspace/datasets",
    #             "./datasets/papers",
    #             save_feat=False)
