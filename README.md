# neural_lyapunov_control

Custom implementation of "Neural Lyapunov Control" originally by Ya-Chien Chang, Nima Roohi, and Sicun Gao. The original work used [dReal](https://github.com/dreal/dreal4), an SMT solver to find counterexamples (falsifier). In my custom implementation, I used sampling based techniques in order to find counterexamples. This is less robust but more computationally efficient.

## How it works

The framework consists of a learner and a falsifier. The learner minimizes the Lyapunov risk to find parameters in both a control function and a neural Lyapunov function. The falsifier takes the learned control function and the neural Lyapunov function from the learner and checks whether there is a state vector violating the Lyapunov conditions (Chang, Roohi, and Gao).


## Requirements

First, create a conda environment with Python 3.8.
```
conda create -n lyapunov python=3.8
```

Activate the environment
```
conda activate lyapunov
```

Install the required packages
```
pip install -r requirements.txt
```

## References
```
@inproceedings{NEURIPS2019_2647c1db,
 author = {Chang, Ya-Chien and Roohi, Nima and Gao, Sicun},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Neural Lyapunov Control},
 url = {https://proceedings.neurips.cc/paper/2019/file/2647c1dba23bc0e0f9cdf75339e120d2-Paper.pdf},
 volume = {32},
 year = {2019}
}
```