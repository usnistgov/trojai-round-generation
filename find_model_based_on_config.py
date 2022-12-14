# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import pandas as pd
import shutil
import random

ifp = '/mnt/scratch/trojai/data/round2/round2-extra'

import package_round_metadata

package_round_metadata.package_metadata(ifp)

ofp = '/mnt/scratch/trojai/data/round2/tmp'
if not os.path.exists(ofp):
    os.makedirs(ofp)

global_results_csv = os.path.join(ifp, 'METADATA.csv')

metadata_df = pd.read_csv(global_results_csv)
models = list(metadata_df['model_name'])

# shuffle the models so I can pick from then sequentially based on the first to match criteria
random.shuffle(models)


# unpack dex factors
tgt_poisoned = True
tgt_class_count = int(9)
# threshold is 15 (10 + 5)
tgt_trigger_type = 'polygon'
tgt_triggered_count = 1
if not tgt_poisoned:
    tgt_triggered_count = 0

# print('Selecting: poisoned {}, class_count {}, type {}, triggered_count {}'.format(tgt_poisoned, tgt_class_count, tgt_trigger_type, tgt_triggered_count))

selected_models = list()

if tgt_poisoned:
    # pick a poisoned model
    found = False
    for model in models:
        df = metadata_df[metadata_df['model_name'] == model]
        if df['poisoned'].to_numpy():
            # pick a model based on DEX
            number_classes = df['number_classes'].to_numpy()
            trigger_type = df['trigger_type'].to_numpy()
            number_triggered_classes = df['number_triggered_classes'].to_numpy()
            if number_triggered_classes > 2:
                # options are 0, 1, 2, N; where N is all
                # so we can unify N down to 3
                number_triggered_classes = 3

            if tgt_class_count == 10:
                if number_classes <= 15:
                    pass  # valid config
                else:
                    continue  # this df is not a valid choice
            if tgt_class_count == 20:
                if number_classes >= 15:
                    pass  # valid config
                else:
                    continue  # this df is not a valid choice

            if trigger_type != tgt_trigger_type:
                continue  # this df is not a valid choice

            if number_triggered_classes != tgt_triggered_count:
                continue  # this df is not a valid choice

            # if we have gotten here, this df represents a valid choice
            selected_models.append(model)
            found = True
            break

    if not found:
        print('Missing: poisoned {}, class_count {}, type {}, triggered_count {}'.format(tgt_poisoned, tgt_class_count, tgt_trigger_type, tgt_triggered_count))
        raise RuntimeError('Ran out of source models')

else:
    # pick a clean model
    found = False
    for model in models:
        df = metadata_df[metadata_df['model_name'] == model]
        if not df['poisoned'].to_numpy():
            selected_models.append(model)
            found = True
            break

    if not found:
        print('Missing: poisoned {}, class_count {}, type {}, triggered_count {}'.format(tgt_poisoned, tgt_class_count, tgt_trigger_type, tgt_triggered_count))
        raise RuntimeError('Ran out of source models')

for m in selected_models:
    print(m)