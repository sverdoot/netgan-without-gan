import abc
import time

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from cell import utils

DEVICE = "cpu"
DTYPE = torch.float32


class Callback(abc.ABC):
    """Abstract Class to customize train loops.
    Applications include logging, printing or the implementation of stopping conditions.
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
    """

    def __init__(self, invoke_every):
        self._training_stopped = False
        self.invoke_every = invoke_every

    def __call__(self, loss, model):
        if model.step % self.invoke_every == 0:
            self.invoke(loss, model)

    def stop_training(self):
        self._training_stopped = True

    @abc.abstractmethod
    def invoke(self):
        """This abstract method is intended to implement the behaviour of classes derived from Callback."""
        pass


class EdgeOverlapCriterion(Callback):
    """Tracks the edge overlap and stops the training if the limit is met.
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
        edge_overlap_limit(float): Stops the training if the models edge overlap reaches this limit.
    """

    def __init__(self, invoke_every, edge_overlap_limit=1.0):
        super().__init__(invoke_every)
        self.edge_overlap_limit = edge_overlap_limit

    def invoke(self, loss, model):
        """Computes the edge overlap and prints the actual step, loss, edge overlap and total time.
        It also stops the training if the computed edge overlap reaches self.edge_overlap_limit.
        Args:
            loss(float): The latest loss value.
            model(Cell): The instance of the model being trained.
        """
        start = time.time()
        model.update_scores_matrix()
        sampled_graph = model.sample_graph()
        overlap = utils.edge_overlap(model.A_sparse, sampled_graph) / model.num_edges
        overlap_time = time.time() - start
        model.total_time += overlap_time

        step_str = f"{model.step:{model.step_str_len}d}"
        print(
            f"Step: {step_str}/{model.steps}",
            f"Loss: {loss:.5f}",
            f"Edge-Overlap: {overlap:.3f}",
            f"Total-Time: {int(model.total_time)}",
        )
        if overlap >= self.edge_overlap_limit:
            self.stop_training()


class LinkPredictionCriterion(Callback):
    """Evaluates the link prediction performance and stops the training if there is no improvement for several steps.
    
    It ensures that the model's score_matrix is set to the score_matrix yielding the best results so far.
    
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
        edge_overlap_limit(float): Stops the training if the models edge overlap reaches this limit.
        val_ones(np.ndarray): Validation ones for link prediction.
        val_zeros(np.ndarray): Validation zeros for link prediction.
        max_patience(int): Maximal number of invokes without improvement of link prediction performance
            until the training is stopped.
    """

    def __init__(self, invoke_every, val_ones, val_zeros, max_patience):
        super().__init__(invoke_every)
        self.val_ones = val_ones
        self.val_zeros = val_zeros
        self.max_patience = max_patience

        self._patience = 0
        self._best_scores_matrix = None
        self._best_link_pred_score = 0.0

    def invoke(self, loss, model):
        """Evaluates the link prediction performance and prints the actual step, loss, edge overlap and total time.
        It also stops the training if there is no improvement for self.max_patience invokes.
        Args:
            loss(float): The latest loss value.
            model(Cell): The instance of the model being trained.
        """
        start = time.time()
        model.update_scores_matrix()
        roc_auc, avg_prec = utils.link_prediction_performance(
            model._scores_matrix, self.val_ones, self.val_zeros
        )

        link_pred_time = time.time() - start
        model.total_time += link_pred_time

        step_str = f"{model.step:{model.step_str_len}d}"
        print(
            f"Step: {step_str}/{model.steps}",
            f"Loss: {loss:.5f}",
            f"ROC-AUC Score: {roc_auc:.3f}",
            f"Average Precision: {avg_prec:.3f}",
            f"Total-Time: {int(model.total_time)}",
        )
        link_pred_score = roc_auc + avg_prec

        if link_pred_score > self._best_link_pred_score:
            self._best_link_pred_score = link_pred_score
            self._best_scores_matrix = model._scores_matrix.copy()
            self._patience = 0

        elif self._patience >= self.max_patience:
            self.stop_training()
        else:
            self._patience += 1
        model._scores_matrix = self._best_scores_matrix


class G_cell(nn.Module):
    def __init__(self, N, H, gamma):
        super().__init__()
        self.W_down = nn.Parameter((
            (gamma * torch.randn(N, H, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))
        self.W_up = nn.Parameter((
            (gamma * torch.randn(H, N, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))

    def add_loss(self):
        return 0

    def forward(self):
        W = torch.mm(self.W_down, self.W_up)
        W -= W.max(dim=-1, keepdims=True)[0]
        return W


class G_cell_local(nn.Module):
    def __init__(self, N, H, gamma):
        super().__init__()
        self.W_down = nn.Parameter((
            (gamma * torch.randn(N, H, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))
        self.W_up = nn.Parameter((
            (gamma * torch.randn(H, N, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))

    def add_loss(self):
        return 0

    def forward(self):
        W = torch.mm(self.W_down, self.W_up)
        W -= W.max(dim=-1, keepdims=True)[0]
        return W


class G_svd(nn.Module):
    def __init__(self, N, H, gamma):
        super().__init__()
        self.N = N
        self.H = H

        self.U = nn.Parameter((
            (gamma * torch.randn(N, H, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))

        self.V = nn.Parameter((
            (gamma * torch.randn(H, N, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))

        self.sigma = nn.Parameter((
            (gamma * torch.randn(H, device=DEVICE, dtype=DTYPE))
                .clone()
                .detach()
                .requires_grad_()
        ))

    def add_loss(self):
        alpha = 1e-4
        return alpha / self.H ** 2 * (torch.norm(self.U.T @ self.U - torch.eye(self.H, self.H), p='fro') ** 2 + \
                                      torch.norm(self.V @ self.V.T - torch.eye(self.H, self.H), p='fro') ** 2)

    def forward(self):
        W = self.U * (self.sigma[None, :]) ** 2
        W = torch.mm(W, self.V)
        W -= W.max(dim=-1, keepdims=True)[0]
        return W


class G_fc(nn.Module):
    def __init__(self, N, H, gamma):
        super().__init__()
        self.N = N
        self.W_down1 = nn.Linear(N, 20 * H, bias=False)
        self.W_down2 = nn.Linear(20 * H, 4 * H, bias=False)
        self.W_down3 = nn.Linear(4 * H, H, bias=False)
        self.W_up1 = nn.Linear(H, N, bias=False)
        # self.W_up2 = nn.Linear(4 * H, N, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform(self.W_down1.weight)
        nn.init.xavier_uniform(self.W_down2.weight)
        nn.init.xavier_uniform(self.W_down3.weight)
        nn.init.xavier_uniform(self.W_up1.weight)
        # nn.init.xavier_uniform(self.W_up2.weight)

    def add_loss(self):
        return 0

    def forward(self):
        vs = torch.eye(self.N, self.N)
        W_down = self.W_down3(F.relu(
            self.W_down2(
                F.relu(
                    self.W_down1(vs)
                ))))
        W = self.W_up1(W_down)

        W -= W.max(dim=-1, keepdims=True)[0]

        return W


class Cell(object):
    """Implements the Cross Entropy Low-rank Logits graph generative model as described our paper.
    
        We approximate the random walk transition matrix of the target graph A over all transition matrices that
        can be expressed by low-rank logits. Approximation is done with respect to the cross-entropy loss. Next, 
        the transition matrix is converted to an edge-independent model, from which the generated graphs are sampled.
        
    Attributes:
        A(torch.tensor): The adjacency matrix representing the target graph.
        A_sparse(sp.csr.csr_matrix): The sparse representation of A.
        H(int): The maximum rank of W.
        loss_fn(function): The loss function minimized during the training process.
        callbacks(list): A list containing instances of classes derived from Callback.
        step(int): Keeps track of the actual training step.
        num_edges(int): The total number of edges in A.
        W_down(torch.tensor): Matrix of shape(N,H) containing optimizable parameters.
        W_up(torch.tensor): Matrix of shape(H,N) containing optimizable parameters.
    """

    def __init__(self, A, H, loss_fn=None, g_type='cell', callbacks=[]):
        self.num_edges = A.sum() / 2
        self.A_sparse = A
        self.A = torch.tensor(A.toarray())
        self.step = 1
        self.callbacks = callbacks
        self._optimizer = None

        N = A.shape[0]
        gamma = np.sqrt(2 / (N + H))

        if g_type == 'cell':
            self.g = G_cell(N, H, gamma)
        elif g_type == 'fc':
            self.g = G_fc(N, H, gamma)
        elif g_type == 'svd':
            self.g = G_svd(N, H, gamma)
        elif g_type == 'local_cell':
            self.g = G_cell_local(N, H, gamma)
        else:
            raise NameError

        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif g_type != 'local_cell':
            self.loss_fn = self.built_in_loss_fn
        else:
            self.loss_fn = self.local_loss
            self.mask = self._compute_mask_for_local_loss(self.A_sparse)
        self.total_time = 0
        self.scores_matrix_needs_update = True

    def __call__(self):
        """Computes the learned random walk transition matrix.
        
        Returns:
            (np.array): The learned random walk transition matrix of shape(N,N)
        """
        return torch.nn.functional.softmax(self.get_W(), dim=-1).detach().numpy()

    def get_W(self):
        """Computes the logits of the learned random walk transition matrix.
        
        Note that we are later interested in row-wise softmax of W.
        Thus, we subtract each row's maximum to improve numerical stability.
        
        Returns:
            W(torch.tensor): Logits of the learned random walk transition matrix of shape(N,N)
        """
        # W = torch.mm(self.W_down, self.W_up)
        # W -= W.max(dim=-1, keepdims=True)[0]
        W = self.g()

        return W

    def built_in_loss_fn(self, W, A, num_edges):
        """Computes the weighted cross-entropy loss in logits with weight matrix.
        Args:
            W(torch.tensor): Logits of learnable (low rank) transition matrix.
            A(torch.tensor): The adjaceny matrix representing the target graph.
            num_edges(int): The total number of edges of the target graph.
            
        Returns:
            (torch.tensor): Loss at logits.
        """
        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        loss = 0.5 * torch.sum(A * (d - W)) / num_edges
        return loss

    def _compute_mask_for_local_loss(self, A_sparse):
        """
        Computes mask (indicator) for local_loss
        Args:
            A_sparse (scipy.sparse.csr_matrix) : adjacency matrix in compressed sparse row format

        Returns:
            mask (torch.tensor): resulting mask tensor
        """
        if not isinstance(A_sparse, scipy.sparse.csr.csr_matrix):
            A_sparse = scipy.sparse.csr_matrix(A_sparse)
        # тут будет долго считать, надо все кратчайшие пути рассчитать, судя по статье
        dists = scipy.sparse.csgraph.shortest_path(csgraph=A_sparse, directed=False)
        # маска вместо индикатора
        return torch.tensor([dists[i, :] < i for i in range(dists.shape[0])], dtype=int)

    def local_loss(self, W, A, num_edges):
        """Computes the LOCAL weighted cross-entropy loss in logits with weight matrix.
        Args:
            W(torch.tensor): Logits of learnable (low rank) transition matrix.
            A(torch.tensor): The adjaceny matrix representing the target graph.
            num_edges(int): The total number of edges of the target graph.

        Returns:
            (torch.tensor): Loss at logits.
        """
        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        # тут хз плюс или минус надо ставить
        return 0.5 * (torch.sum(A * (d - W)) / num_edges + torch.sum(self.mask * A * (d - W)) / num_edges)

    def _closure(self):
        W = self.get_W()
        loss = self.loss_fn(W=W, A=self.A, num_edges=self.num_edges) + self.g.add_loss()
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    def _train_step(self):
        """Performs and times one optimization step."""
        time_start = time.time()
        loss = self._optimizer.step(self._closure)
        time_end = time.time()
        return loss.item(), (time_end - time_start)

    def train(self, steps, optimizer_fn, optimizer_args, EO_criterion=None):
        """Starts the train loop.
        """
        self._optimizer = optimizer_fn(self.g.parameters(), **optimizer_args)
        self.steps = steps
        self.step_str_len = len(str(steps))
        self.scores_matrix_needs_update = True
        stop = False
        for self.step in range(self.step, steps + self.step):
            loss, time = self._train_step()
            self.total_time += time
            for callback in self.callbacks:
                callback(loss=loss, model=self)
                stop = stop or callback._training_stopped
            if stop:
                break

    def update_scores_matrix(self):
        """Updates the score matrix according to W."""
        self._scores_matrix = utils.scores_matrix_from_transition_matrix(
            transition_matrix=self(), symmetric=True
        )
        self.scores_matrix_needs_update = False

    def sample_graph(self):
        """Samples a graph from the learned parameters W.
        
        Edges are sampled independently from the score maxtrix.
        
        Returns:
            sampled_graph(sp.csr.csr_matrix): A synthetic graph generated by the model.
        """
        if self.scores_matrix_needs_update:
            self.update_scores_matrix()

        sampled_graph = utils.graph_from_scores(self._scores_matrix, self.num_edges)
        return sampled_graph
