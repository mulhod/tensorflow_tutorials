#!/bin/zsh

THIS_DIR="$(dirname "$(readlink -f $0)")"

# Create Conda environment with TensorFlow, etc., installed
echo "Creating Conda environment to use for the TensorFlow tutorials, etc."
CONDA_ENV="${THIS_DIR}/tf_env"
[[ -e "${CONDA_ENV}" ]] && { conda env remove --prefix "${CONDA_ENV}" }
conda install --prefix "${CONDA_ENV}" --yes -m --file --copy "${THIS_DIR}"/conda_requirements.txt
echo "Created TensorFlow Conda environment in \"${CONDA_ENV}\"."
echo "Run \"source activate ${CONDA_ENV}\" to use the Conda environment" \
     "and \"source deactivate\" to get out of the environment."

# Clone TensorFlow repository and check out the "v0.10.0" branch
echo "Cloning the TensorFlow repository and checking out the \"v0.10.0\"" \
     "branch and making symlinks to the examples and examples/tutorials" \
     "directories..."
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v0.10.0
ln -s "${THIS_DIR}/tensorflow/tensorflow/examples" "${THIS_DIR}/tensorflow_repository_examples"
ln -s "${THIS_DIR}/tensorflow/tensorflow/examples/tutorials" "${THIS_DIR}/tensorflow_repository_tutorials"

# Start the Jupyter notebook server
PORT=8889
echo "Starting a Jupyter notebook server on port ${PORT}..."
source activate "${CONDA_ENV}/bin"
"${CONDA_ENV}"/bin/jupyter notebook --no-browser --port ${PORT} &
echo "Started a Jupyter notebook server on port ${PORT}. Navigate to" \
     "localhost:8889 in a browser to start using the tutorial notebooks."
