import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp
from scipy.sparse import load_npz
import pandas as pd
import random
import torch

from cell.utils import (
    load_npz, 
    largest_connected_components, 
    edge_overlap, 
    train_val_test_split_adjacency,
    link_prediction_performance
    )
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion, StatisticCollector
from cell.graph_statistics import compute_graph_statistics

EDGE_OVERLAP_LIMIT = {
    'CORA-ML' : 0.7, 
    'Citeseer' : 0.8,
    'PolBlogs': 0.41,
    'RT-GOP': 0.7
}
MAX_STEPS = 400


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/cora_ml.npz')
    parser.add_argument('--data_name', type=str, default='cora_ml')
    parser.add_argument('--statistics_step', type=int, default=10)
    parser.add_argument('--number_of_samples', type=int, default=5)
    parser.add_argument('--H', type=int, default=9)
    parser.add_argument('--g_type', type=str, default='all')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--graphic_mode', type=str, default='overlap', choices=['overlap', 'iterations'])
    parser.add_argument('--fig_path', type=str, default=None)
    parser.add_argument('--table_path', type=str, default=None)
    parser.add_argument('--eo_limit', type=float, default=None)
    parser.add_argument('--criterion', type=str, choices=['eo', 'val'], default='eo')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    return args


def main(args):
    _A_obs, _X_obs, _z_obs = load_npz(args.data_path)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    _A_obs = _A_obs - sp.eye(_A_obs.shape[0], _A_obs.shape[0])
    _A_obs[_A_obs < 0] = 0
    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]

    val_share = 0.05
    test_share = 0.1
    seed = 42

    train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(_A_obs, 
                                                                                            val_share, 
                                                                                            test_share, 
                                                                                            seed, 
                                                                                            undirected=True, 
                                                                                            connected=True, 
                                                                                            asserts=True)
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    if args.eo_limit is not None:
        edge_overlap_limit = args.eo_limit 
    elif args.data_name in EDGE_OVERLAP_LIMIT.keys():
        edge_overlap_limit = EDGE_OVERLAP_LIMIT[args.data_name]
    else:
        edge_overlap_limit = 0.6

    training_stat = dict()
    
    if args.g_type != 'all':
        training_stat[args.g_type] = list()
    else:
        training_stat = {i: list() for i in ['cell', 'fc', 'svd']}
    
    df = pd.DataFrame()
    for model_name in list(training_stat.keys()):    
        print(f"'{model_name.upper()}' Approach")
        optimizer_args = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if model_name == 'fc':
            optimizer_args['weight_decay'] = 1e-4
        if args.criterion == 'eo':
            callbacks = [EdgeOverlapCriterion(invoke_every=10, 
                                            edge_overlap_limit=edge_overlap_limit)]
        elif args.criterion == 'val':
            callbacks = [LinkPredictionCriterion(invoke_every=2,
                                                val_ones=val_ones,
                                                val_zeros=val_zeros,
                                                max_patience=5)]

        if args.fig_path is not None:
            invoke_every = args.statistics_step 
        else:
            invoke_every = MAX_STEPS + 1
        stat_collector = StatisticCollector(invoke_every, _A_obs, test_ones, test_zeros, graphic_mode=args.graphic_mode, n_samples=args.number_of_samples)
        callbacks.append(stat_collector)

        model = Cell(A=train_graph, 
                    H=args.H, 
                    g_type=model_name, 
                    callbacks=callbacks)

        model.train(steps=MAX_STEPS,
                optimizer_fn=torch.optim.Adam,
                optimizer_args=optimizer_args)

        stat_collector.invoke(None, model)
        training_stat[model_name] = stat_collector.training_stat
        stats = training_stat[model_name][-1]['stats']
        stat_df = pd.DataFrame({k: [s[k] for s in stats] for k in stats[0].keys()})
        stat_df = stat_df.mean()
        df[model_name] = stat_df.T

    original_stat = compute_graph_statistics(_A_obs)
    df[args.data_name] = list(original_stat.values()) + [1, 1, 1]
    if args.table_path is not None:
        df.to_csv(args.table_path)

    if args.fig_path is not None:
        fig, axs = plt.subplots(3, 3, figsize=(15,9))
        fig.suptitle(args.data_name, fontsize=18)
        for stat_id, (stat_name, stat) in enumerate(
            list(zip(['Max.Degree','Assortativity', 'Power law exp.', 'Rel. edge distr. entr', 
            'Clustering coeff.', 'Gini coeff.','Wedge count', 'Triangle count', 'Square count'], 
            ['d_max', 'assortativity', 'power_law_exp', 'rel_edge_distr_entropy', 'clustering_coefficient', 'gini',
            'wedge_count', 'triangle_count', 'square_count']))):

            axs[stat_id // 3, stat_id % 3].set_ylabel(stat_name, fontsize=18)
            if args.graphic_mode == 'overlap':
                axs[stat_id // 3, stat_id % 3].set_xlabel('Edge overlap (in %)', fontsize=13)
            else:
                axs[stat_id // 3, stat_id % 3].set_xlabel('Iterations', fontsize=13)
            axs[stat_id // 3, stat_id % 3].axhline(y=original_stat[stat], color='g', linestyle='--', label='target')
            for model_name, model_statistic in training_stat.items():
                if args.graphic_mode == 'overlap':
                    xs = [100 * i['overlap'] for i in model_statistic]
                else:
                    xs = [i['iteration'] for i in model_statistic]
                    
                axs[stat_id // 3, stat_id % 3].errorbar(xs, 
                                                        [np.mean([j[stat] for j in i['stats']]) for i in model_statistic],
                                                        [np.std([j[stat] for j in i['stats']]) for i in model_statistic],
                                                        ls='none',
                                                        fmt='.',
                                                        label=model_name
                                                        )

        axLine, axLabel = axs[stat_id // 3, stat_id % 3].get_legend_handles_labels()
        fig.legend(axLine, axLabel, loc = 'center right', fontsize=15)
        fig.tight_layout()
        if args.fig_path is not None:
            plt.savefig(args.fig_path)
        plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)
