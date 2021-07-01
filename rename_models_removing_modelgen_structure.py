# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import glob
import shutil

fps = list()
fps.append('/scratch/trojai/data/round2/round2-train')
fps.append('/scratch/trojai/data/round2/round2-test')
fps.append('/scratch/trojai/data/round2/round2-holdout')
fps.append('/scratch/trojai/data/round2/models')

for ifp in fps:
    print(ifp)

    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    models.sort()

    for model in models:
        if not os.path.exists(os.path.join(ifp, model, 'model')):
            continue

        print(model)
        model_filepath = glob.glob(os.path.join(ifp, model, 'model', 'DataParallel*.pt.1'))
        if len(model_filepath) != 1:
            raise RuntimeError('more than one model file')
        model_filepath = model_filepath[0]

        stats_filepath = glob.glob(os.path.join(ifp, model, 'model', 'DataParallel*.pt.1.stats.detailed.csv'))
        if len(stats_filepath) != 1:
            raise RuntimeError('more than one detailed stats file')
        stats_filepath = stats_filepath[0]

        json_filepath = glob.glob(os.path.join(ifp, model, 'model', 'DataParallel*.pt.1.stats.json'))
        if len(json_filepath) != 1:
            raise RuntimeError('more than one json file')
        json_filepath = json_filepath[0]

        dest = os.path.join(ifp, model, 'model.pt')
        shutil.move(model_filepath, dest)
        dest = os.path.join(ifp, model, 'model_detailed_stats.csv')
        shutil.move(stats_filepath, dest)
        dest = os.path.join(ifp, model, 'model_stats.json')
        shutil.move(json_filepath, dest)

        shutil.rmtree(os.path.join(ifp, model, 'model'))
