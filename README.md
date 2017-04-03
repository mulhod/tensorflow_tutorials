# TensorFlow Tutorials

## Requirements
- Miniconda must be installed. If it is not installed, go [here](http://conda.pydata.org/miniconda.html), download the installation script for your system (it doesn't matter if it's Python 2 or 3), and then run it with (the exact command will change depending on the installation script's name): `bash Miniconda3-latest-Linux-x86_64.sh -b -f -p $INSTALL_LOCATION` (where `$INSTALL_LOCATION` refers to a place that you can install Miniconda without root privileges, etc., could be your home directory or anything you prefer).

## Setup
- In order to set up the environment and start a Jupyter notebook server on port 8889, run [setup.sh](./setup.sh): `zsh setup.sh`.
- This will create a Conda environment in `tf_env`, which will have all packages defined in [`conda_requirements.txt`](./conda_requirements.txt) installed in it, including TensorFlow version 0.12.1.
- It will also make a clone of the TensorFlow GitHub repository just for the purposes of being able to refer to data, etc., that exists as part of the repository.
- Lastly, it will launch a Jupyter notebook server on port 8889. If the setup script was already run once and all you want to do is start the Jupyter notebook server, the setup script can be run with the `--start_jupyter_only` command-line flag.

## [Tutorials Home](https://www.tensorflow.org/tutorials/)
- All tutorials are in the directories within the [`tutorial_notebooks`](./tutorial_notebooks) directory.
* [MNIST for ML Beginners](tutorial_notebooks/MNIST_for_ML_Beginners/MNIST_for_ML_Beginners.ipynb)
* [TensorFlow Mechanics 101](tutotial_notebooks/TensorFlow_Mechanics_101/TensorFlow_Mechanics_101.ipynb)
