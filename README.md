# NetGAN-without-GAN
This repository contains Pytorch implementation of the paper [NetGAN without GAN:
From Random Walks to Low-Rank Approximations](https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/RensburgLuxburg_Netgan_without_Gan2020.pdf) and additional experiments of low-rank approximation of transformation matrix of random walks through graph.

## Table of content:
  - [Table of content:](#table-of-content)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Usage](#usage)
  - [References](#references)


## Directory structure


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

generate $\varepsilon$-neighbourhood graph:

```bat
python data/eps_neighbour.py \
  --save_name eps_neighbour_graph
```

## References
