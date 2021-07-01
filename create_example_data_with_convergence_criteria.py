# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import shutil
import json

import rebuild_single_dataset
import inference

image_format = 'png'
# ifp = '/mnt/scratch/trojai/data/round2/round2-holdout-dataset'
ifp = '/mnt/scratch/trojai/data/round2/round2-checkpoint-clean-models-trained-100-epochs'
output_clean_data = True
if output_clean_data:
    accuracy_result_fn = 'example-accuracy.csv'
    example_data_fn = 'example_data'
else:
    accuracy_result_fn = 'poisoned-example-accuracy.csv'
    example_data_fn = 'poisoned_example_data'


models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
models.sort()

fail_list = list()

for model in models:
    example_accuracy = 0
    cur_fp = os.path.join(ifp, model, accuracy_result_fn)
    if os.path.exists(cur_fp):
        with open(cur_fp, 'r') as example_fh:
            example_accuracy = float(example_fh.readline())

    if example_accuracy < 99:
        print(model)

        i = 0
        while example_accuracy < 99:
            i = i + 1
            if i > 10:
                print('Attempted rebuilding example data 10 times for {} and failed'.format(model))
                fail_list.append(model)
                break
            if os.path.exists(os.path.join(ifp, model, example_data_fn)):
                shutil.rmtree(os.path.join(ifp, model, example_data_fn))
            if os.path.exists(cur_fp):
                os.remove(cur_fp)

            if output_clean_data:
                rebuild_single_dataset.clean(os.path.join(ifp, model))
            else:
                rebuild_single_dataset.poisoned(os.path.join(ifp, model))

            config_fp = os.path.join(ifp, model, 'config.json')
            with open(config_fp, 'r') as fp:
                config = json.load(fp)

            image_folder = os.path.join(ifp, model, example_data_fn)
            model_filepath = os.path.join(ifp, model, 'model.pt')
            if not os.path.exists(model_filepath):
                import glob
                model_filepath = glob.glob(os.path.join(ifp, model, 'model', 'DataParallel_*.pt.1'))[0]
            example_accuracy, per_img_logits = inference.inference_get_model_accuracy(image_folder, image_format, model_filepath, config['TRIGGER_TARGET_CLASS'])
            print('  Try {}: {} = {}'.format(i, accuracy_result_fn, example_accuracy))

            with open(os.path.join(ifp, model, accuracy_result_fn), 'w') as fh:
                fh.write('{}'.format(example_accuracy))

            # TODO test this
            with open(os.path.join(ifp, model, accuracy_result_fn.replace('accuracy','logits')), 'w') as fh:
                fh.write('Example, Logits\n')
                for k in per_img_logits.keys():
                    fh.write('{}, {}\n'.format(k, per_img_logits[k]))



if len(fail_list) > 0:
    print('The following models failed to have the required accuracy')
    for m in fail_list:
        print(m)