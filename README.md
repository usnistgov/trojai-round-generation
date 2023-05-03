# Getting Started

This codebase supports generating trojaned models for vision based tasks. It currently can generate image-classification and object-detection trojaned models. Extending to support semantic segmentation is on the TODO list.

Run `main_demo.py` to explore the codebase capabilities before diving into the full extensible capability with `main.py`.

Before running the code, you will need to setup the git submodules, and build a conda environment with the required libraries. 


# git Submodule

This repo needs a submodule from: https://github.com/usnistgov/pytorch_utils

Submodule management documentation can be found at https://git-scm.com/book/en/v2/Git-Tools-Submodules

## Initialize Submodule

`git submodule init`

This command should be run to initialize the submodule in the repo after cloning.

## Pull changes from Submodule (after init)

`git submodule update --remote`

This command should be run each time you want to pull the latest version of the submodule. 


## Conda Env Setup

```
# Install anaconda3
# https://www.anaconda.com/distribution/
# or miniconda3
# https://docs.conda.io/en/latest/miniconda.html
# conda config --set auto_activate_base false

conda update -n base -c defaults conda

# create a virtual environment to stuff all these packages into
conda create -n trojai python=3.9 -y

# activate the virtual environment
conda activate trojai

# install pytorch (best done through conda to handle cuda dependencies)
conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# IMPORTANT: Must install ImageMagick in order to use the wand library for instagram triggers
# See: https://docs.wand-py.org/en/0.6.5/guide/install.html
conda install -c conda-forge imagemagick -y

conda install pandas scikit-learn psutil  -y

pip install jsonpickle matplotlib pycocotools opencv-python imgaug imagecorruptions albumentations blend_modes wand torchmetrics timm transformers

conda env export > conda_environment.yml
```


