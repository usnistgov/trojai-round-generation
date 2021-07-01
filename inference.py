# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import skimage.io
import torch


def inference_get_model_accuracy(image_folder, image_format, model_filepath, poisoned_gt_class = None):
    if image_format.startswith('.'):
        image_format = image_format[1:]
    imgs = [fn for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    model = torch.load(model_filepath).cuda()

    logits = dict()

    nb_correct = 0
    for img_fn in imgs:
        toks = img_fn.split('_')
        ground_truth_class = int(toks[1])
        if poisoned_gt_class is not None:
            ground_truth_class = poisoned_gt_class

        # # read the image
        img = skimage.io.imread(os.path.join(image_folder, img_fn))

        # perform center crop to what the CNN is expecting 224x224
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy + 224, dx:dx + 224, :]

        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)
        # convert image to a gpu tensor
        batch_data = torch.FloatTensor(img)

        batch_data = batch_data.cuda()

        # inference the image
        logits = model(batch_data).cpu().detach().numpy()
        pred = np.argmax(logits)

        logits[img_fn] = logits

        if pred == ground_truth_class:
            nb_correct += 1

    accuracy = 100.0 * float(nb_correct) / len(imgs)
    return accuracy, logits


