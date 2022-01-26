#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --nodelist=pn116125
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=r8
#SBATCH -o log-%N.%j.out
#SBATCH --time=72:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

# export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /mnt/isgnas/home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate round8

START_RUN=$1
MODELS_PER_JOB=$2

# control where huggingface caches models
#export TRANSFORMERS_CACHE=".model_cache"

# control where huggingface caches datasets
#export HF_DATASETS_CACHE=".dataset_cache"

# Control the CPU core count outside of slurm (train script respects the slurm env variable)
#export SLURM_CPUS_PER_TASK="10"

root_source_dataset_directory="/mnt/isgnas/home/mmajursk/trojai/source_data/qa"

root_output_directory="/mnt/isgnas/home/mmajursk/trojai/r8/models"
if ! [ -d ${root_output_directory} ]; then
    mkdir ${root_output_directory}
fi


python train_model.py --output-filepath=${root_output_directory} --start-model-number="$START_RUN" --number-models-to-build="$MODELS_PER_JOB" --source-datasets-filepath="$root_source_dataset_directory"
	

