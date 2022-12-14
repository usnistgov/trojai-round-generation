# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch
import torchvision.models.detection
from transformers import DetrConfig, DetrForObjectDetection
import timm

import round_config


def load_model(config: round_config.RoundConfig):
    model_architecture_name = config.model_architecture.split(':')[1]
    if config.task_type == round_config.OBJ:
        if 'ssd300_vgg16' in model_architecture_name:
            # This loads the arch and a pre-trained backbone, but the object detector is randomly initialized
            model = torchvision.models.detection.ssd300_vgg16(progress=True, trainable_backbone_layers=5, num_classes=config.number_classes)
        elif 'fasterrcnn_resnet50_fpn_v2' in model_architecture_name:
            # This loads the arch and a pre-trained backbone, but the object detector is randomly initialized
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(progress=True, num_classes=config.number_classes, trainable_backbone_layers=5)
        elif 'detr' in model_architecture_name:
            # This loads the arch and a pre-trained backbone, but the object detector is randomly initialized
            config = DetrConfig(num_labels=config.number_classes)
            model = DetrForObjectDetection(config)
        else:
            raise RuntimeError('Unknown model architecture name: {}'.format(model_architecture_name))
    elif config.task_type == round_config.CLASS:
        model_weight = config.model_weight
        if model_weight == 'none':
            model_weight = None

        if "resnet" in model_architecture_name:
            model = torch.hub.load("pytorch/vision:v0.13.0", model_architecture_name, progress=True, weights=model_weight)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, config.number_classes)
        # elif "alexnet" in model_architecture_name:
        #     num_ftrs = model.classifier[6].in_features
        #     model.classifier[6] = torch.nn.Linear(num_ftrs, config.number_classes)
        # elif "vgg" in model_architecture_name:
        #     num_ftrs = model.classifier[6].in_features
        #     model.classifier[6] = torch.nn.Linear(num_ftrs, config.number_classes)
        elif "squeezenet" in model_architecture_name:
            model = torch.hub.load("pytorch/vision:v0.13.0", model_architecture_name, progress=True, weights=model_weight)
            model.classifier[1] = torch.nn.Conv2d(512, config.number_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = config.number_classes
        elif "densenet" in model_architecture_name:
            model = torch.hub.load("pytorch/vision:v0.13.0", model_architecture_name, progress=True, weights=model_weight)
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, config.number_classes)
        elif "mobilenet" in model_architecture_name:
            model = torch.hub.load("pytorch/vision:v0.13.0", model_architecture_name, progress=True, weights=model_weight)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, config.number_classes)
        elif 'vit' in model_architecture_name:
            model = timm.create_model(model_architecture_name, img_size=round_config.RoundConfig.BASE_IMAGE_SIZE, pretrained=True)
            num_ftrs = model.head.in_features
            model.head = torch.nn.Linear(num_ftrs, config.number_classes)

        # elif "inception" in model_architecture_name:
        #     # Handle the auxilary net
        #     num_ftrs = model.AuxLogits.fc.in_features
        #     model.AuxLogits.fc = torch.nn.Linear(num_ftrs, config.number_classes)
        #     # Handle the primary net
        #     num_ftrs = model.fc.in_features
        #     model.fc = torch.nn.Linear(num_ftrs, config.number_classes)
        else:
            raise RuntimeError('Unknown model architecture name: {}'.format(model_architecture_name))
    else:
        raise RuntimeError("Invalid task type: {}".format(config.task_type))

    return model
