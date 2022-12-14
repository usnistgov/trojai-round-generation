## Join the Challenge

To join the TrojAI challenge: <https://pages.nist.gov/trojai/docs/accounts.html#request-account>

## Overview
This is the TrojAI code used to construct the image-classification-feb2021 round of the TrojAI challenge. 

Each round of the TrojAI challenge has its own branch where the train/test/holdout dataset generation code lives.

Check out the TrojAI leaderboard here: <https://pages.nist.gov/trojai/> or the documentation here: <https://pages.nist.gov/trojai/docs/index.html>.

## Generating Data

This repository contains code to both bulk train AI models as part of building the released TrojAI datasets, as well as post process the trained models coming off the compute cluster(s).

The main entry point is a sbatch (SLURM) script titled `create_single_dataset_sbatch.sh`.

This script does 2 things, first it builds a config file, then it trains a model instantiating that config file into a trained AI.

If this script is run on a remote machine, you will need to download the resulting trained AI models to your machine before you can post-process them to filter out non-converged models and building example data.

Given a folder of trained models, running `move_completed_models.py` will find those models which have completed and generated an output pytorch model file. This handles missing model files seamlessly in case the job failed or was preempted on the cluster. 

After running `move_completed_models.py` you should build the example data using: `create_example_data_with_convergence_criteria.py`. This script builds example images drawn from the same distribution which build the train/test data. It then inferences these images using the specified model and computes the accuracy of the model on the example data. A convergence criteria is specified to ensure that the examples built meet that accuracy criteria, or are printed to the terminal as 'failed'.

With the example data constructed you can now use the script `subset_converged_models.py` to select a set of N converged models from the set of all models that have been built so far (the script selects from models existing in the input folder). This script selects models based on the Design of Experiment (DEX) factors to ensure the sampling of the model configurations is uniform across the DEX factors. 



## Acknowledgements
This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

## Disclaimer

NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
