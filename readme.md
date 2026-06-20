# Representation Learning via Composite Subspace Flow Matching

This repository contains the PyTorch implementation for the master's thesis: **"Representation Learning via Composite Subspace Flow Matching"**.

In this work, we address the fundamental trade-offs between joint-embedding architectures (semantics) and generative models (spatial topology). We propose **Composite Subspace Flow Matching (CSFM)**, a framework that enforces differential survival rates for macroscopic content and microscopic style, decoupling the optimization of representations from low-level pixel variations.

Our codebase adapts and builds upon the foundational frameworks provided by [traj_drl](https://github.com/sarthmit/traj_drl) and [score_sde](https://github.com/yang-song/score_sde).

## Codebase Overview

For researchers and reviewers looking for our specific contributions, please refer to the following files:
* `target_flow_lib.py`: Contains the physics engine for our Decoupled Optimal Transport and CSFM degradation paths.
* `losses.py`: Implements our core objectives, including the CSFM flow matching loss with Asymmetric Gradient Routing, alongside baselines (CDAE, LeJEA, AE).
* `run_lib.py`: The main orchestration file managing the training loops, EMA updates, and distributed optimization.


## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
pip install -r requirements.txt
```


## Usage

### Full Pipeline (Train, Eval, Sample)

To run a complete CSFM model experiment (which automatically cycles through training, evaluation, and sampling based on your config intervals), execute:

Bash

```
python main.py --mode=all --config=experiments/csfm_example/config_csfm.py --workdir=experiments/csfm_example
```

### Isolated Evaluation or Sampling

If you already have a trained checkpoint and want to evaluate features or generate new samples, you can run those modes in isolation:

Bash

```
# For feature evaluation (saves to features.h5)
python main.py --mode=eval --config=experiments/csfm_example/config_csfm.py --workdir=experiments/csfm_example --checkpoint=path/to/checkpoint.pth

# For generating samples
python main.py --mode=sample --config=experiments/csfm_example/config_csfm.py --workdir=experiments/csfm_example --checkpoint=path/to/checkpoint.pth
```

*All outputs, including model checkpoints, HDF5 embeddings, and generated image grids, will be saved automatically to the directory specified in `--workdir`.*