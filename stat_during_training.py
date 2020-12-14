import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp
from scipy.sparse import load_npz
import pandas as pd

import torch
from cell.utils import load_npz, largest_connected_components, edge_overlap
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion
from cell.graph_statistics import compute_graph_statistics

EDGE_OVERLAP_LIMIT = {
    'cora_ml' : 0.7, 
    'citeseer' : 0.8,
    'polblogs': 0.7
}
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/cora_ml.npz')
    parser.add_argument('--data_name', type=str, default='cora_ml')
    parser.add_argument('--statistics_step', type=int, default=10)
    parser.add_argument('--number_of_samples', type=int, default=5)
    parser.add_argument('--g_type', type=str, default='cell')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--save_path', type=str, default='img/stat.png')
    args = parser.parse_args()
    return args


def main(args):
    _A_obs, _X_obs, _z_obs = load_npz(args.data_path)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]

    model = Cell(A=_A_obs, H=9, g_type=args.g_type, callbacks=[EdgeOverlapCriterion(invoke_every=10, edge_overlap_limit=.7)]) 
    optimizer_args = {'lr': args.lr, 'weight_decay': args.weight_decay} 
    model._optimizer  = torch.optim.Adam(model.g.parameters(), **optimizer_args)    
    if args.data_name in EDGE_OVERLAP_LIMIT.keys():
        edge_overlap_limit = EDGE_OVERLAP_LIMIT[args.data_name]
    else:
        edge_overlap_limit = 0.6

    iter = 0
    current_overlap = 0.0  
    training_stat = list()
    while current_overlap < edge_overlap_limit:
        iter += 1
        loss, time = model._train_step()
        model.update_scores_matrix()
        if iter % args.statistics_step == 0:
            generated_graphs = [model.sample_graph() for _ in range(args.number_of_samples)]
            current_overlap = np.mean([edge_overlap(model.A_sparse, gg) / model.num_edges for gg in generated_graphs])
            stats = [compute_graph_statistics(gg) for gg in generated_graphs]
            training_stat.append({'overlap': current_overlap, 'stats': stats})
            print(f"Iteration: {iter}; loss: {loss}; overlap: {current_overlap}")

    original_stat = compute_graph_statistics(model.A_sparse)

    fig, axs = plt.subplots(3, 3, figsize=(12,12))
    fig.suptitle(args.data_name, fontsize=18)
    for stat_id, (stat_name, stat) in enumerate(
        list(zip(['Max.Degree','Assortativity', 'Power law exp.', 'Rel. edge distr. entr', 
        'Clustering coeff.', 'Gini coeff.','Wedge count', 'Triangle count', 'Square count'], 
        ['d_max', 'assortativity', 'power_law_exp', 'rel_edge_distr_entropy', 'clustering_coefficient', 'gini',
        'wedge_count', 'triangle_count', 'square_count']))):

        axs[stat_id // 3, stat_id % 3].set_ylabel(stat_name, fontsize=18)
        axs[stat_id // 3, stat_id % 3].set_xlabel('Edge overlap (in %)', fontsize=13)
        axs[stat_id // 3, stat_id % 3].axhline(y=original_stat[stat], color='g', linestyle='--', label='target')
        axs[stat_id // 3, stat_id % 3].errorbar([100 * i['overlap'] for i in training_stat], 
                                                [np.mean([j[stat] for j in i['stats']]) for i in training_stat],
                                                [np.std([j[stat] for j in i['stats']]) for i in training_stat],
                                                ls='none',
                                                fmt='.',
                                                label=args.g_type
                                                )

    axLine, axLabel = axs[stat_id // 3, stat_id % 3].get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc = 'center right', fontsize=15)
    fig.tight_layout()
    if args.save_path is not None:
        plt.savefig(args.save_path)
    plt.close()



if __name__ == '__main__':
    args = parse_arguments()
    main(args)
