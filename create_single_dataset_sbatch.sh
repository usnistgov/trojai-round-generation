#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=trojai
#SBATCH -o log-%N.%j.out
#SBATCH --time=24:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.



# limit the script to only the GPUs selected
#gpuId=${1}
#export CUDA_DEVICE_ORDER="PCI_BUS_ID"
#export CUDA_VISIBLE_DEVICES=${gpuId}
#echo "Using GPU(s): $gpuId"

source /home/mmajursk/anaconda3/etc/profile.d/conda.sh
conda activate round2

root_output_directory="/wrk/mmajursk/round2"
if ! [ -d ${root_output_directory} ]; then
  mkdir ${root_output_directory}
fi

for number in {1000..1999}; do
    cd "/home/mmajursk/round2"
    printf -v numStr "%08d" $number

    scratch_dir="/scratch/${SLURM_JOB_ID}/id-${numStr}"
    output_directory="$root_output_directory/id-${numStr}"

    if [ -d ${output_directory} ]; then
      continue
    fi

    mkdir ${output_directory}
    mkdir ${scratch_dir}

    foregrounds_filepath="/wrk/mmajursk/source_data/foregrounds"
    backgrounds_filepath="/wrk/mmajursk/source_data/backgrounds"

    if [ $((${RANDOM}%2)) -eq 0 ]  # select triggered or not at random
    then
      echo "Generating a triggered model"
      python create_single_dataset.py --poison --foreground_images_filepath=${foregrounds_filepath} --background_images_filepath=${backgrounds_filepath} --output_filepath=${scratch_dir}
    else
      echo "Generating a clean model"
      python create_single_dataset.py --foreground_images_filepath=${foregrounds_filepath} --background_images_filepath=${backgrounds_filepath} --output_filepath=${scratch_dir}
    fi
    exitCode=$?
    if ! [ $exitCode -eq 0 ]; then
        rm -rf ${scratch_dir}
        exit 1
    fi

    python train_model.py --dataset-filepath=${scratch_dir}

    cd ${scratch_dir}
    rm -rf *.lmdb # delete the image data
    # compress the example images to make copy faster
    tar -czf example_data.tar.gz example_data
    # delete the example images, keeping the tar ball
    rm -rf example_data
    rm train.csv
    rm test-*.csv
    cd ..
    mv ${scratch_dir}/* ${output_directory}/

    # cleanup scratch space, this deletes the model without copy if a non-zero exit code is encountered
    rm -rf ${scratch_dir}/*
    if ! [ $exitCode -eq 0 ]; then
            exit 1
    fi
done