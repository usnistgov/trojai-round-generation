# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# Install anaconda3
# https://www.anaconda.com/distribution/
# conda config --set auto_activate_base false

conda update -n base -c defaults conda

# create a virtual environment to stuff all these packages into
conda create -n round11 python=3.8 -y

# activate the virtual environment
conda activate round11

# install pytorch (best done through conda to handle cuda dependencies)
conda install pytorch=1.12 torchvision=0.13 cudatoolkit=11.3 -c pytorch

# IMPORTANT: Must install ImageMagick in order to use the wand library for instagram triggers
# See: https://docs.wand-py.org/en/0.6.5/guide/install.html
conda install -c conda-forge imagemagick

conda install pandas

pip install timm jsonpickle matplotlib sklearn seqeval pycocotools opencv-python imgaug imagecorruptions psutil albumentations transformers blend_modes wand torchmetrics

conda env export > conda_environment.yml


# Fix pending bug in torchmetrics
# https://github.com/Lightning-AI/metrics/issues/1184
# and
# https://github.com/pytorch/pytorch/issues/82457
# 1) open mean_ap.py file from torchmetrics
# 2) goto function __evaluate_image_gt_no_preds and edit line 505
# 3) "dtScores": torch.zeros(nb_det, dtype=torch.bool, device=self.device),
#     should become
#    "dtScores": torch.zeros(nb_det, dtype=torch.float32, device=self.device),
# this can be done by
# cd ~/anaconda3/envs/round11/lib/python3.8/site-packages/torchmetrics/detection
# vim mean_ap.py
# scroll to line 505, and make the relevant edit.
# if you re-install the torchmetrics package, you will need to redo this change


