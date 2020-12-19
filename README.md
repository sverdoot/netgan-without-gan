# NetGAN-without-GAN
This repository contains Pytorch implementation of the paper [NetGAN without GAN:
From Random Walks to Low-Rank Approximations](https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/RensburgLuxburg_Netgan_without_Gan2020.pdf) and additional experiments of low-rank approximation of transformation matrix of random walks through graph.

## Table of content:
  - [Some results](#some-results)
  - [Repository structure](#repository-structure)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Usage](#usage)
  - [References](#references)


## Some results

   |         |   d_max |   d_min |     d |    LCC |   wedge_cnt |   claw_cnt |   triangle_cnt |   square_cnt |   power_law_exp |   gini |   edge_distr_ent |   assortat |   cluster_coef |   cpl |   ROC-AUC |   AVG-PREC |    EO |
|---------|---------|---------|-------|--------|---------------|--------------|------------------|----------------|-----------------|--------|--------------------------|-----------------|--------------------------|-------|-----------|------------|-------|
| CELL    |   187.4 |       1 | 4.828 | 2798.8 |       80068.6 |  1.61e+06 |           1389.4 |         6619.6 |           1.821 |  0.45  |                    0.95  |          -0.076 |                    0.052 | 5.243 |     0.945 |      0.953 | 0.56  |
| NonlinCELL      |   178.2 |       1 | 4.828 | 2798.4 |       84728.2 |  1.67e+06 |           1769.6 |        10316.2 |           1.831 |  0.46  |                    0.947 |          -0.078 |                    0.063 | 5.368 |     0.962 |      0.965 | 0.545 |
| SVD     |   181   |       1 | 4.828 | 2809.2 |       75677.4 |  1.41e+06 |           1014.4 |         4709.6 |           1.806 |  0.437 |                    0.953 |          -0.081 |                    0.04  | 5.051 |     0.951 |      0.957 | 0.513 |
| CORA-ML |   246   |       1 | 5.68  | 2810   |      137719   |  3.93e+06 |           5247   |        34507   |           1.767 |  0.496 |                    0.939 |          -0.045 |                    0.114 | 5.271 |     1     |      1     | 1     |


## Repository structure

The structure of the project is flat, no informative nested directories. 

```cell``` contains implementation of main methods

```data``` contains used datasets and script for generating epsilon neighbourhood graph

```img``` contains visaulizations of evolutions of methods and vizualizations of generated graphs

```logs``` contains tables with results

```notebooks/demo.ipynb``` contains a range of experiments on different datasets

```notebooks/eps_neighbour_graph.ipynb``` contains experiment with generated epsilon neughbourhood graph

```notebooks/tabs.ipynb``` contains latex tables generator 


## Installation

```bat
git clone https://github.com/sverdoot/netgan-without-gan
```

## Setup

```bat
cd [VENV]
virtualenv netgan-without-gan
source netgan-without-gan/bin/activate
```

back in repository dir:
```bat
pip install -r requirements.txt
pip install -e .
```


## Usage

run experiments:

```bat
python run_experiments.py \
  --data_path data/cora_ml.npz \
  --data_name CORA-ML \
  --table_path logs/cora_ml.csv \
  --H 9 \
  --seed 42 \
  --eo_limit 0.5
```

```bat
python run_experiments.py \
  --data_path data/citeseer.npz \
  --data_name Citeseer \
  --table_path logs/citeseer.csv \
  --H 9 \
  --seed 42 \
  --eo_limit 0.5
```

```bat
python run_experiments.py \
  --data_path data/polblogs.npz \
  --data_name PolBlogs \
  --table_path logs/polblogs.csv \
  --H 9 \
  --seed 42 \
  --eo_limit 0.5
```

```bat
python run_experiments.py \
  --data_path data/rt_gop.npz \
  --data_name RT-GOP \
  --table_path logs/rt_gop.csv \
  --H 9 \
  --seed 42 \
  --eo_limit 0.5
```

generate epsilon-neighbourhood graph:

```bat
python data/eps_neighbour.py \
  --save_name eps_neighbour_graph
```

## References
