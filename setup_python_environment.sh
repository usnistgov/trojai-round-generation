#!/bin/bash
# ********************************
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: You cannot run this script, you need to walk through it manurally

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ********************************

# Install anaconda3
# https://www.anaconda.com/distribution/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
# bash Anaconda3-2019.10-Linux-x86_64.sh

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.




# create a virtual environment to stuff all these packages into
conda create --name round2 python=3.7 -y
# activate the virtual environment
conda activate round2


conda install pillow
# install imagemagick 7
conda install -c conda-forge imagemagick

# install all other non-trojai requirements
pip install --upgrade lmdb

# install trojai repos
git clone https://bitbucket.xrcs.jhuapl.edu/scm/troj/albumentations.git
cd albumentations
pip install -e .

cd ..
git clone https://bitbucket.xrcs.jhuapl.edu/scm/troj/trojai.git
cd trojai
pip install -e .

cd ..
git clone https://bitbucket.xrcs.jhuapl.edu/scm/troj/trojai_private.git
cd trojai_private
pip install -e .

# install pytorch (best done through conda to handle cuda dependencies)
# TrojAI will install pytorch to 1.5, so we need to overwrite with 1.7
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly

