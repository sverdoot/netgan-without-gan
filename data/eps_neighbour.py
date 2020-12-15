import torch
import scipy.sparse as sp
import numpy as np
import random
import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--x_min', type=float, default=0)
    parser.add_argument('--x_max', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--nvert', type=int, default=1000)
    parser.add_argument('--max_nvert', type=int, default=100)
    parser.add_argument('--save_name', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    # adj = sp.csr_matrix((args.nvert, args.nvert))
    coos = torch.rand(args.nvert, args.dim, dtype=torch.float32)
    sq_dists = ((coos[None, :, :] - coos[:, None, :])**2).sum(-1)
    adj = sp.csr_matrix(sq_dists**.5 < args.eps)
    adj = adj - sp.eye(args.nvert)
    adj[adj < 0] = 0
    loader = dict()
    loader['adj_data'] = adj.data
    loader['adj_indices'] = adj.indices
    loader['adj_indptr'] = adj.indptr
    loader['adj_shape'] = adj.shape
    loader['coos'] = coos.numpy()
    np.savez(Path(Path(__file__).parent, args.save_name), **loader)

