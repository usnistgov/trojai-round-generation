#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=r12
#SBATCH -o log-%N.%j.out
#SBATCH --time=96:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


source /mnt/isgnas/home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate round11

START_RUN=$1
MODELS_PER_JOB=$2



root_output_directory="/mnt/isgnas/home/mmajursk/trojai/r12/models"
if ! [ -d ${root_output_directory} ]; then
    mkdir ${root_output_directory}
fi


source_dataset_directory="/mnt/isgnas/home/mmajursk/trojai/r12/source_data"


INDEX=$START_RUN


for i in $(seq $MODELS_PER_JOB); do
 python main.py --output-filepath=${root_output_directory} --model-number="$INDEX" --dataset-dirpath="$source_dataset_directory"
 INDEX=$((INDEX+1))
done


#
## start the first 2 trains right away
#python main.py --output-filepath=${root_output_directory} --model-number="$INDEX" --dataset-dirpath="$source_dataset_directory" &
#INDEX=$((INDEX+1))
#sleep 1
#python main.py --output-filepath=${root_output_directory} --model-number="$INDEX" --dataset-dirpath="$source_dataset_directory" &
#INDEX=$((INDEX+1))
#sleep 1
#
## loop over the remainder of the count to be generated, and when one of the two running jobs completes, launch a new one
#for i in $(seq $MODELS_PER_JOB); do
# wait -n
# python main.py --output-filepath=${root_output_directory} --model-number="$INDEX" --dataset-dirpath="$source_dataset_directory" &
# INDEX=$((INDEX+1))
# sleep 1
#done

## wait for all of the runs to complete before exiting
#wait