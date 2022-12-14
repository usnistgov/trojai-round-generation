# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import json
import torch

import models


def wrap_stock_pytorch_model(model_filepath: str, overwrite:bool = False):
    """
    Function to convert (if required) the stock pytorch model format into the NIST-ified version from models.py whose forward pass returns both the logits as well as the loss.

    Args:
        model_filepath: filepath to the "id-xxxxxxxx" folder to convert (if conversion is required).

    """

    #print("Starting conversion of {}".format(os.path.basename(model_filepath)))
    config_fp = os.path.join(model_filepath, 'config.json')
    with open(config_fp) as json_file:
        config_dict = json.load(json_file)
        if 'py/state' in config_dict.keys():
            config_dict = config_dict['py/state']

    # create a baseline NIST-ified version of the model in memory
    if config_dict['model_architecture'] == 'fasterrcnn':
        net = models.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=5)
    elif config_dict['model_architecture'] == 'ssd':
        net = models.ssd300_vgg16(pretrained=True, trainable_backbone_layers=5)
    elif config_dict['model_architecture'] == 'detr':
        net = models.detr_model()
    else:
        raise RuntimeError("Invalid Model Architecture: {}".format(config_dict['model_architecture']))

    # create the filepath to the stock pytorch version of the model on disk (early models in the round10 generation pipeline were stock pytorch, later NIST wrapped those models to produce both a loss and logits during the forward pass).
    model_pt_filepath = os.path.join(model_filepath, 'model.pt')

    # load the model from disk using the normal method
    disk_model = torch.load(model_pt_filepath)

    # test if this model has already been wrapped into the NIST models.py version
    if not hasattr(disk_model, 'prepare_inputs'):
        print("Starting conversion of {}".format(os.path.basename(model_filepath)))
        print("  wrapping model.")
        # this model on disk has not yet been wrapped into the NIST-ified version

        # get the disk model's state_dict
        state_dict = disk_model.state_dict()
        # load that state dict into the baseline NIST-ified version of the model
        net.load_state_dict(state_dict)

        if not overwrite:
            # create a new filepath to avoid overwriting the stock pytorch version of the model on disk
            model_pt_filepath = os.path.join(model_filepath, 'model_wrapped.pt')

        print("  saving wrapped model to {}".format(model_pt_filepath))
        torch.save(net, model_pt_filepath)
    # else:
    #     print("  model is already in correct format.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to convert stock PyTorch Round10 models into the NIST-ified version in models.py')
    parser.add_argument('--dataset_dirpath', type=str, required=True, help="Filepath to the folder/directory where TrojAI dataset is containing the list of \"id-xxxxxxxx\" folders.")
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the \"model.pt\" files, or save the wrapped models into a new file \"model_wrapped.pt\".')

    args = parser.parse_args()

    fldrs = [fn for fn in os.listdir(args.dataset_dirpath) if fn.startswith('id-')]
    fldrs.sort()

    for fldr in fldrs:
        model_filepath = os.path.join(args.dataset_dirpath, fldr)
        wrap_stock_pytorch_model(model_filepath, args.overwrite)
