import torch
import torch.nn as nn
from dgl import backend as F
from shmtensor import capi
import weakref


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor = torch.zeros(num_embeddings,
                                  embedding_dim,
                                  dtype=torch.float32).uniform_(-1, 1)
        self.ids_traces = []
        self.emb_traces = []
        self.name = "embedding" + str(id(self))

        capi.pin_memory(self.tensor)
        weakref.finalize(self, capi.unpin_memory, self.tensor)

    def forward(self, ids):
        # emb = self.tensor[ids].cuda()
        emb = capi.uva_fetch(self.tensor, ids)

        if not self.training:
            return emb

        emb.requires_grad_(True)
        emb.retain_grad()

        if F.is_recording():
            emb = F.attach_grad(emb)
            self.ids_traces.append(ids.to('cuda', non_blocking=True))
            self.emb_traces.append(emb.to('cuda', non_blocking=True))

        return emb


def unique_grads(idx, grad):
    grad_indices, inverse, grad_cnt = torch.unique(idx,
                                                   return_inverse=True,
                                                   return_counts=True)
    grad_values = torch.zeros((grad_indices.shape[0], grad.shape[1]),
                              device=grad.device)
    grad_values.index_add_(0, inverse, grad)
    grad_values = grad_values / grad_cnt.unsqueeze(1)
    return grad_indices, grad_values


class SparseAdam(nn.Module):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-08):
        super().__init__()

        self._params = params
        self._eps = eps
        self._beta1, self._beta2 = betas
        self._lr = lr
        self._state = {}

        for param in params:
            state_step = torch.zeros(param.num_embeddings, dtype=torch.float32)
            state_mem = torch.zeros(
                (param.num_embeddings, param.embedding_dim),
                dtype=torch.float32)
            state_power = torch.zeros(
                (param.num_embeddings, param.embedding_dim),
                dtype=torch.float32)

            self._state[param.name] = (state_step, state_mem, state_power)

            capi.pin_memory(self._state[param.name][0])
            capi.pin_memory(self._state[param.name][1])
            capi.pin_memory(self._state[param.name][2])

            weakref.finalize(self, capi.unpin_memory,
                             self._state[param.name][0])
            weakref.finalize(self, capi.unpin_memory,
                             self._state[param.name][1])
            weakref.finalize(self, capi.unpin_memory,
                             self._state[param.name][2])

    def update(self, idx, grad, emb):
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        clr = self._lr
        state = self._state[emb.name]

        # unique first
        grad_indices, grad_values = unique_grads(idx, grad)

        state_step = capi.uva_fetch(state[0], grad_indices)
        orig_mem = capi.uva_fetch(state[1], grad_indices)
        orig_power = capi.uva_fetch(state[2], grad_indices)

        state_step = state_step.cuda()
        orig_mem = orig_mem.cuda()
        orig_power = orig_power.cuda()

        state_step = state_step + 1

        grad_mem = grad_values
        grad_power = grad_values * grad_values
        update_mem = beta1 * orig_mem + (1.0 - beta1) * grad_mem
        update_power = beta2 * orig_power + (1.0 - beta2) * grad_power

        #state[0][state_idx] = state_step.cpu()
        #state[1][state_idx] = update_mem.cpu()
        #state[2][state_idx] = update_power.cpu()

        capi.uva_set(state[0], grad_indices, state_step)
        capi.uva_set(state[1], grad_indices, update_mem)
        capi.uva_set(state[2], grad_indices, update_power)

        update_mem_corr = update_mem / (1.0 - torch.pow(
            torch.tensor(beta1, device='cuda'), state_step)).unsqueeze(1)
        update_power_corr = update_power / (1.0 - torch.pow(
            torch.tensor(beta2, device='cuda'), state_step)).unsqueeze(1)
        std_values = clr * update_mem_corr / (torch.sqrt(update_power_corr) +
                                              eps)

        # emb.tensor[state_idx] -= std_values.cpu()

        update_emb = capi.uva_fetch(emb.tensor, grad_indices)
        update_emb = update_emb - std_values
        capi.uva_set(emb.tensor, grad_indices, update_emb)

    def zero_grad(self):
        self._clean_grad = True

    def step(self):
        with torch.no_grad():
            for param in self._params:
                ids = torch.cat(param.ids_traces)
                grads = torch.cat([i.grad.data for i in param.emb_traces])
                self.update(ids, grads, param)

            param.ids_traces.clear()
            param.emb_traces.clear()


class SparseAdagrad(nn.Module):

    def __init__(self, params, lr, eps=1e-10):
        super().__init__()

        self._params = params
        self._eps = eps
        self._lr = lr
        self._state = {}

        for param in params:
            state = torch.zeros((param.num_embeddings, param.embedding_dim),
                                dtype=torch.float32)
            self._state[param.name] = state

    def update(self, idx, grad, emb):
        eps = self._eps
        clr = self._lr
        state = self._state[emb.name]

        # unique first
        grad_indices, grad_values = unique_grads(idx, grad)

        state_idx = grad_indices.cpu()
        state_value = state[state_idx]
        state_value = state_value.cuda()

        grad_sum = grad_values * grad_values
        state_value += grad_sum

        state[state_idx] = state_value.cpu()

        std_values = state_value.sqrt_().add_(eps)
        tmp = clr * grad_values / std_values

        emb.tensor[state_idx] -= tmp.cpu()

    def zero_grad(self):
        self._clean_grad = True

    def step(self):
        with torch.no_grad():
            for param in self._params:
                ids = torch.cat(param.ids_traces)
                grads = torch.cat([i.grad.data for i in param.emb_traces])
                self.update(ids, grads, param)

            param.ids_traces.clear()
            param.emb_traces.clear()
