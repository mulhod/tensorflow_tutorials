#!/bin/zsh

THIS_DIR="$(dirname "$(readlink -f $0)")"
CONDA_ENV="${THIS_DIR}/tf_env"
[[ -e "${CONDA_ENV}" ]] && { conda env remove --prefix "${CONDA_ENV}" }
conda install --prefix "${CONDA_ENV}" --yes -m --file --copy "${THIS_DIR}"/conda_requirements.txt
source activate "${CONDA_ENV}/bin"
