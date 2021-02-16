#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=trojai
#SBATCH -o log-%N.%j.out
#SBATCH --time=24:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

printf -v numStr "%08d" ${1}
echo "id-$numStr"

source /home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate round4

root_output_directory="/wrk/mmajursk/round4"
# make the root output directory
if ! [ -d ${root_output_directory} ]; then
    mkdir ${root_output_directory}
fi

output_directory="$root_output_directory/id-${numStr}"

# if the output directory already exists, exit
if [ -d ${output_directory} ]; then
  exit 0
fi

# make the output directory
mkdir ${output_directory}

# foregrounds is a folder of images to select a set of N 'classes' from
foregrounds_filepath="/wrk/mmajursk/source_data/image-classification/foregrounds"
# backgrounds is a folder of folders, each sub-folder contains many images to be used as backgrounds.
backgrounds_filepath="/wrk/mmajursk/source_data/image-classification/backgrounds"

if [ $((${RANDOM}%2)) -eq 0 ]  # select triggered or not at random
then
    echo "Generating a triggered model"
    python create_config.py --poison --foreground_images_filepath=${foregrounds_filepath} --background_images_filepath=${backgrounds_filepath} --output_filepath=${output_directory}
else
    echo "Generating a clean model"
    python create_config.py --foreground_images_filepath=${foregrounds_filepath} --background_images_filepath=${backgrounds_filepath} --output_filepath=${output_directory}
fi

# train the model based on the config file
python train_model.py --filepath=${output_directory}