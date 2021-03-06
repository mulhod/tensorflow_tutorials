#!/bin/zsh

# Usage: ./setup.sh[ --start_jupyter_only]

START_JUPYTER_ONLY=0
if [[ $# -gt 0 ]]; then
    if [[ $# -gt 1 || $1 != "--start_jupyter_only" ]]; then
        echo "ERROR: Got unexpected command-line argument(s). Only" \
             "expecting one at most (--start_jupter_only). Exiting."
        exit 1
    else
        START_JUPYTER_ONLY=1
    fi
fi

if [[ $OSTYPE == "darwin16.0" ]]; then
    THIS_DIR=$(python -c "from __future__ import print_function; import os.path; print(os.path.dirname(os.path.realpath(\"$0\")))")
else
    THIS_DIR=$(dirname $(readlink -f $0))
fi
CONDA_ENV="${THIS_DIR}/tf_env"

if [[ "${START_JUPYTER_ONLY}" == 0 ]]; then

    # Create Conda environment with TensorFlow, etc., installed
    echo "Creating Conda environment to use for the TensorFlow tutorials, etc."
    [[ -e "${CONDA_ENV}" ]] && { conda env remove --prefix "${CONDA_ENV}" }
    conda create --prefix "${CONDA_ENV}" --yes --copy --mkdir python=3.5
    conda install --channel conda-forge --prefix "${CONDA_ENV}" --yes --copy \
        --file "${THIS_DIR}"/conda_requirements.txt
    echo "Created TensorFlow Conda environment in \"${CONDA_ENV}\"."
    echo "Run \"source activate ${CONDA_ENV}\" to use the Conda environment" \
         "and \"source deactivate\" to get out of the environment."

    # Clone TensorFlow repository and check out the latest commit for the
    # "v0.12.1" release
    echo "Cloning the TensorFlow repository and checking out the latest" \
         "commit for the v0.12.1 release and making symlinks to the examples" \
         "and examples/tutorials directories..."
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    git checkout 4d924e796368163
    cd ..
    ln -s "${THIS_DIR}/tensorflow/tensorflow/examples" \
          "${THIS_DIR}/tensorflow_repository_examples"
    ln -s "${THIS_DIR}/tensorflow/tensorflow/examples/tutorials" \
          "${THIS_DIR}/tensorflow_repository_tutorials"

fi

# Start the Jupyter notebook server
PORT=8889
echo "Starting a Jupyter notebook server on port ${PORT}..."
source activate "${CONDA_ENV}"
"${CONDA_ENV}"/bin/jupyter notebook --no-browser --port ${PORT} &
echo "Started a Jupyter notebook server on port ${PORT}. Navigate to" \
     "localhost:8889 in a browser to start using the tutorial notebooks."

