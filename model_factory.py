import torch
import torch.nn
import torchvision.models.detection
import timm
from transformers import DetrConfig, DetrForObjectDetection


def load_model_classification(config):
    """
    Function loads the requested model architecture. The model_weight attribute is passed along to the torch.hug.load to load the requested weights.

    Returns: torch.nn.Module model instance.
    """

    if "resnet" in config.model_architecture.value:
        if config.model_weight.value is None:
            # load randomly initialized models
            if config.model_architecture.value == 'resnet18':
                model = torchvision.models.resnet18(weights=None)
            elif config.model_architecture.value == 'resnet34':
                model = torchvision.models.resnet34(weights=None)
            elif config.model_architecture.value == 'resnet50':
                model = torchvision.models.resnet50(weights=None)
            elif config.model_architecture.value == 'resnet101':
                model = torchvision.models.resnet101(weights=None)
            else:
                raise NotImplementedError('Unknown model architecture name: {}'.format(config.model_architecture.value))
        else:
            # load the requested weights by name
            model = torch.hub.load("pytorch/vision", config.model_architecture.value, weights=config.model_weight.value)

        # adjust the number of classes
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, config.number_classes.value)

    elif "squeezenet" in config.model_architecture.value:
        if config.model_weight.value is None:
            # load randomly initialized models
            if config.model_architecture.value == 'squeezenet1_0':
                model = torchvision.models.squeezenet1_0(weights=None)
            elif config.model_architecture.value == 'squeezenet1_1':
                model = torchvision.models.squeezenet1_1(weights=None)
            else:
                raise NotImplementedError('Unknown model architecture name: {}'.format(config.model_architecture.value))
        else:
            model = torch.hub.load("pytorch/vision", config.model_architecture, progress=True, weights=config.model_weight.value)

        # adjust the number of classes
        model.classifier[1] = torch.nn.Conv2d(512, config.number_classes.value, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = config.number_classes.value

    elif "densenet" in config.model_architecture.value:
        if config.model_weight.value is None:
            # load randomly initialized models
            if config.model_architecture.value == 'densenet121':
                model = torchvision.models.densenet121(weights=None)
            elif config.model_architecture.value == 'densenet161':
                model = torchvision.models.densenet161(weights=None)
            elif config.model_architecture.value == 'densenet169':
                model = torchvision.models.densenet169(weights=None)
            elif config.model_architecture.value == 'densenet201':
                model = torchvision.models.densenet201(weights=None)
            else:
                raise NotImplementedError('Unknown model architecture name: {}'.format(config.model_architecture.value))
        else:
            model = torch.hub.load("pytorch/vision:v0.13.0", config.model_architecture.value, progress=True, weights=config.model_weight.value)

        # adjust the number of classes
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, config.number_classes.value)

    elif "mobilenet" in config.model_architecture.value:
        if config.model_weight.value is None:
            # load randomly initialized models
            if config.model_architecture.value == 'mobilenet_v2':
                model = torchvision.models.mobilenet_v2(weights=None)
            else:
                raise NotImplementedError('Unknown model architecture name: {}'.format(config.model_architecture.value))
        else:
            model = torch.hub.load("pytorch/vision:v0.13.0", config.model_architecture.value, progress=True, weights=config.model_weight.value)

        # adjust the number of classes
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, config.number_classes.value)

    elif 'vit' in config.model_architecture.value:
        if config.model_weight.value is None:
            model = timm.create_model(config.model_architecture.value, img_size=config.BASE_IMAGE_SIZE, pretrained=False)
        else:
            model = timm.create_model(config.model_architecture.value, img_size=config.BASE_IMAGE_SIZE, pretrained=True)

        # adjust the number of classes
        num_ftrs = model.head.in_features
        model.head = torch.nn.Linear(num_ftrs, config.number_classes.value)
    else:
        raise NotImplementedError('Unknown model architecture name: {}'.format(config.model_architecture.value))

    return model


def load_model_object_detection(config):
    """
    Function loads the requested model architecture. The model_weight attribute is passed along to the torch.hug.load to load the requested weights.

    Returns: torch.nn.Module model instance.
    """

    number_classes = config.number_classes.value + 1  # obj det class ids start at 1, so total class count needs to match

    if 'ssd300_vgg16' in config.model_architecture.value:
        if config.model_weight.value is None:
            model = torchvision.models.detection.ssd300_vgg16(progress=True, trainable_backbone_layers=5, num_classes=number_classes)
        else:
            # This loads the arch and a pre-trained backbone, but the object detector final classification layer is randomly initialized
            checkpoint = torchvision.models.detection.ssd300_vgg16(progress=True, trainable_backbone_layers=5, weights=config.model_weight.value).state_dict()
            # remove the class weights, which need to match the number of classes
            del checkpoint["head.classification_head.module_list.0.weight"]
            del checkpoint["head.classification_head.module_list.0.bias"]
            del checkpoint["head.classification_head.module_list.1.weight"]
            del checkpoint["head.classification_head.module_list.1.bias"]
            del checkpoint["head.classification_head.module_list.2.weight"]
            del checkpoint["head.classification_head.module_list.2.bias"]
            del checkpoint["head.classification_head.module_list.3.weight"]
            del checkpoint["head.classification_head.module_list.3.bias"]
            del checkpoint["head.classification_head.module_list.4.weight"]
            del checkpoint["head.classification_head.module_list.4.bias"]
            del checkpoint["head.classification_head.module_list.5.weight"]
            del checkpoint["head.classification_head.module_list.5.bias"]

            model = torchvision.models.detection.ssd300_vgg16(progress=True, trainable_backbone_layers=5, num_classes=number_classes)
            # overwrites the models wights (except for the final class layer) with the state dict values.
            model.load_state_dict(checkpoint, strict=False)  # strict false to allow missing keys (the class head)

    elif 'fasterrcnn_resnet50_fpn_v2' in config.model_architecture.value:
        if config.model_weight.value is None:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(progress=True, num_classes=number_classes, trainable_backbone_layers=5)
        else:
            # This loads the arch and a pre-trained backbone, but the object detector final classification layer is randomly initialized
            checkpoint = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(progress=True, weights=config.model_weight.value, trainable_backbone_layers=5).state_dict()
            # remove the class weights, which need to match the number of classes
            del checkpoint['roi_heads.box_predictor.cls_score.weight']
            del checkpoint['roi_heads.box_predictor.cls_score.bias']
            del checkpoint['roi_heads.box_predictor.bbox_pred.weight']
            del checkpoint['roi_heads.box_predictor.bbox_pred.bias']

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(progress=True, num_classes=number_classes, trainable_backbone_layers=5)
            # overwrites the models wights (except for the final class layer) with the state dict values.
            model.load_state_dict(checkpoint, strict=False)  # strict false to allow missing keys (the class head)

    elif 'detr' in config.model_architecture.value:
        if config.model_weight.value is None:
            model = DetrForObjectDetection(DetrConfig(num_labels=number_classes, backbone='resnet50'))
        elif str(config.model_weight.value).lower() == 'default':
            # This loads the arch and a pre-trained backbone, but the object detector final classification layer is randomly initialized
            checkpoint = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").state_dict()
            # remove the class weights, which need to match the number of classes
            del checkpoint["class_labels_classifier.weight"]
            del checkpoint["class_labels_classifier.bias"]

            model = DetrForObjectDetection(DetrConfig(num_labels=number_classes, backbone='resnet50'))
            # overwrites the models wights (except for the final class layer) with the state dict values.
            model.load_state_dict(checkpoint, strict=False)  # strict false to allow missing keys (the class head)
        else:
            msg = 'Invalid pretrained weights value = {}, cannot load pre-trained model.'.format(config.model_weight.value)
            raise RuntimeError(msg)
    else:
        raise RuntimeError('Unknown model architecture name: {}'.format(config.model_architecture.value))

    return model