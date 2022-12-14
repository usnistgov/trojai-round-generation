# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import typing
import torchmetrics
import torchmetrics.detection.mean_ap

from train import TrojanModelTrainer

import round_config
import torch
import logging

logger = logging.getLogger()


class PerSampleAccuracy(torchmetrics.Accuracy):
    """
    Version of the Accuracy metric which calculates per-sample accuracy. This needs to be its own class because TorchMetrics cannot have 2 of the same metric in the MetricCollection.
    """
    def __init__(self, **kwargs: typing.Any):
        # remove the 2 args we are specifying the values of
        if 'average' in kwargs:
            kwargs.pop('average')
        if 'reduce' in kwargs:
            kwargs.pop('reduce')
        super().__init__(average='none', reduce='samples', **kwargs)

class PerClassAccuracy(torchmetrics.Accuracy):
    """
    Version of the Accuracy metric which calculates per-sample accuracy. This needs to be its own class because TorchMetrics cannot have 2 of the same metric in the MetricCollection.
    """
    def __init__(self, num_classes:int, **kwargs: typing.Any):
        # remove the 2 args we are specifying the values of
        if 'average' in kwargs:
            kwargs.pop('average')
        if 'reduce' in kwargs:
            kwargs.pop('reduce')
        if 'num_classes' in kwargs:
            kwargs.pop('num_classes')
        super().__init__(average='none', reduce='macro', num_classes=num_classes, **kwargs)


class TrojanClassificationTrainer(TrojanModelTrainer):

    # allow default collate function to work
    collate_fn = None


    def __init__(self, config: round_config.RoundConfig):

        self.loss_function = torch.nn.CrossEntropyLoss()

        accuracy = torchmetrics.Accuracy()
        per_sample_accuracy = PerSampleAccuracy()
        per_class_accuracy = PerClassAccuracy(num_classes=config.number_classes)
        metrics = torchmetrics.MetricCollection([accuracy, per_sample_accuracy, per_class_accuracy])

        super().__init__(config, metrics)

    def model_forward(self, model, images, targets, only_return_loss=False):
        # model.train()  # leave model in train mode, even for evaluation to bypass an error with pytorch AMP and the ViT function _native_multi_head_attention
        logits = model(images)
        batch_train_loss = self.loss_function(logits, targets.long())
        if only_return_loss:
            return batch_train_loss
        else:
            return batch_train_loss, logits

    def get_image_targets_on_gpu(self, tensor_dict):
        images = tensor_dict[0].to(self.device)
        targets = tensor_dict[1].to(self.device)

        return images, targets


class TrojanObjectDetectionTrainer(TrojanModelTrainer):
    def __init__(self, config: round_config.RoundConfig):

        mAP = torchmetrics.detection.mean_ap.MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

        metrics = torchmetrics.MetricCollection([mAP])

        super().__init__(config, metrics)

    @staticmethod
    def collate_fn(batch):
        # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
        return tuple(zip(*batch))

    def model_forward(self, model, images, targets, only_return_loss=False):
        base_mode = model.training
        model.train()
        outputs = model(images, targets)
        losses = sum(loss for loss in outputs.values())
        if only_return_loss:
            model.train(base_mode)  # restore the model back to the original training status
            return losses

        # TODO this double inference horrifies me, but its simpler than completely re-writing/wrapping the individual model architectures to return both loss and detections.
        model.eval()
        # for SSD/FasterRCNN/DETR detection boxes are in [xyxy] format
        detections = model(images, targets)
        model.train(base_mode)  # restore the model back to the original training status
        return losses, detections

    def get_image_targets_on_gpu(self, tensor_dict):
        # tensor_dict[0] is a tuple of batch_size images
        # tensor_dict[1] is a tuple of batch_size target dicts
        for tgt in tensor_dict[1]:
            tgt['boxes'] = tgt['boxes'].type(torch.FloatTensor)
            tgt['labels'] = tgt['labels'].type(torch.LongTensor)
        images = [img.to(self.device) for img in tensor_dict[0]]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in tensor_dict[1]]

        return images, targets


class TrojanObjectDetectionDetrTrainer(TrojanObjectDetectionTrainer):
    def __init__(self, config: round_config.RoundConfig):
        super().__init__(config)

    def model_forward(self, model, images, targets, only_return_loss=False):
        base_mode = model.training
        model.train()

        # organize the targets based on what DETR needs
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L1380
        #  labels (`List[Dict]` of len `(batch_size,)`, *optional*):
        #             Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        #             following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        #             respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        #             in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        block_input = torch.stack(images)
        # The boxes should be in [x0, y0, x1, y1] (corner) format.
        for t in targets:
            # add the required re-naming of the labels for DETR
            t['class_labels'] = t['labels']
            # normalize the boxes from [0, BASE_IMAGE_SIZE] to [0, 1]
            t['boxes'][:, 0::2] = t['boxes'][:, 0::2] / round_config.RoundConfig.BASE_IMAGE_SIZE
            t['boxes'][:, 1::2] = t['boxes'][:, 1::2] / round_config.RoundConfig.BASE_IMAGE_SIZE

        outputs = model(block_input, labels=targets)
        losses = outputs.loss
        if only_return_loss:
            model.train(base_mode)  # restore the model back to the original training status
            return losses

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        def center_to_corners_format(x):
            """
            Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
            (x_0, y_0, x_1, y_1).
            """
            x_c, y_c, w, h = x.unbind(-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=-1)

        # convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h = round_config.RoundConfig.BASE_IMAGE_SIZE * torch.ones(len(images))
        img_w = round_config.RoundConfig.BASE_IMAGE_SIZE * torch.ones(len(images))
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(self.device)
        boxes = boxes * scale_fct[:, None, :]

        detections = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        model.train(base_mode)  # restore the model back to the original training status
        return losses, detections
