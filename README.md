# GNNPointerNet

This repository contains code for training a GNNPointerNet model 
to generate **2D Delaunay-like meshes** from point coordinates.

It is basically an immitation of Delaunay Meshing. The input is a sequence 2D points and the output is a sequences of triangles. 

## Project Poster
[![Poster Preview](assets/poster_preview.png)](assets/MeshingPoster.pdf)

## Repo Structure
- `data/` 
    - `data.py` the script that contains all the tools to generate the dataset.
    - `generate_datasets.py` run this script to generate one train and one test dataset. 
- `gnnpointernet/` — main source code
    - `losses/loss_functions.py` the script that contains the training loss and evaluation metrics. 
    - `models/model.py` the script that contains the model structure and definitions. 
    - `utils/util.py` the script that contains the plotting tool. 
- `experiments/` training and evaluation code 
    - `train.py` the script that builds and trains and log a GNNPointerNet model.
    - `evaluate.py` the script that loads the logged models and evaluates the model. 
- `notebooks/` Jupyter notebooks of my scratch codes and others. 
- `assets/`
    - `MeshingPoster.pdf` my poster for this project
    - `poster_preview.png` a preview png image of the poster. 

## Getting Started
```bash
pip install -r requirements.txt
python -m gnnpointernet.train --config configs/default.yaml
```

# GNNPointerNet

This repository contains code for training a Graph Neural Network (GNN) model 
to generate **2D Delaunay-like meshes** from point coordinates.

## Structure
- `data/` — raw and processed data, plus loaders
- `gnnpointernet/` — main source code
- `configs/` — experiment configs
- `experiments/` — logs, checkpoints, outputs
- `notebooks/` — Jupyter notebooks for exploration
- `tests/` — unit tests

## Getting Started
```bash
pip install -r requirements.txt
python -m gnnpointernet.train --config configs/default.yaml
```

