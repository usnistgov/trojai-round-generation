# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import shutil

def worker(ifp, root_ofp):
    ofp_models = os.path.join(root_ofp, 'models')
    ofp_gt = os.path.join(root_ofp, 'groundtruth')

    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    models.sort()

    if not os.path.exists(root_ofp):
        os.makedirs(root_ofp)
    if not os.path.exists(ofp_gt):
        os.makedirs(ofp_gt)
    if not os.path.exists(ofp_models):
        os.makedirs(ofp_models)

    for m in models:
        if os.path.exists(os.path.join(ofp_models, m)):
            continue

        os.makedirs(os.path.join(ofp_models, m))

        pt_model_filepath = os.path.join(ifp, m, 'model.pt')
        model_filepath = os.path.join(ofp_models, m, 'model.pt')
        shutil.copyfile(pt_model_filepath, model_filepath)

        if not os.path.exists(os.path.join(ofp_gt, m)):
            os.makedirs(os.path.join(ofp_gt, m))
        shutil.copyfile(os.path.join(ifp, m, 'ground_truth.csv'), os.path.join(ofp_gt, m, 'ground_truth.csv'))

        shutil.copytree(os.path.join(ifp, m, 'example_data'), os.path.join(ofp_models, m, 'example_data'))


# ifp = '/mnt/scratch/trojai/data/round2/round2-test-dataset'
# root_ofp = '/mnt/scratch/trojai/data/round2/test_server/es-dataset'
# worker(ifp, root_ofp)

ifp = '/mnt/scratch/trojai/data/round2/round2-holdout-dataset'
root_ofp = '/mnt/scratch/trojai/data/round2/test_server/holdout-dataset'
worker(ifp, root_ofp)
