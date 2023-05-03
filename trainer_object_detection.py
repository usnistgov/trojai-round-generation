import warnings
import copy
import logging
import time
import psutil
import numpy as np
import torch
import torchmetrics
import torchmetrics.detection.mean_ap

# local imports
import base_config
import trainer
import dataset
import detection_data
import bbox_utils
import utils


class ObjectDetectionTrainer(trainer.ModelTrainer):
    def __init__(self, config: base_config.Config):
        mAP = torchmetrics.detection.mean_ap.MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

        metrics = torchmetrics.MetricCollection([mAP])

        super().__init__(config, metrics)

    @staticmethod
    def collate_fn(batch):
        # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
        return tuple(zip(*batch))

    @staticmethod
    def set_BN_to_eval(model: torch.nn.Module):
        # if the model should be in eval mode (but can't be to capture loss), turn off updates to BatchNorm running stats
        for module in model.modules():
            # print(module)
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def model_forward(self, model: torch.nn.Module, images, targets, only_return_loss: bool = False):
        base_mode = model.training
        if base_mode != True:
            model.train()
            # if the model should be in eval mode, turn off updates to BatchNorm running stats
            ObjectDetectionTrainer.set_BN_to_eval(model)

        outputs = model(images, targets)
        losses = sum(loss for loss in outputs.values())
        if only_return_loss:
            if base_mode != model.training:
                model.train(base_mode)  # restore the model back to the original training status
            return losses

        # this double inference horrifies me, but its simpler than completely re-writing/wrapping the individual model architectures to return both loss and detections.
        model.eval()
        # for SSD/FasterRCNN/DETR detection boxes are in [xyxy] format
        detections = model(images, targets)
        if base_mode != model.training:
            model.train(base_mode)  # restore the model back to the original training status
        return losses, detections

    def get_image_targets_on_gpu(self, tensor_dict: dict[any]):
        # tensor_dict[0] is a tuple of batch_size images
        # tensor_dict[1] is a tuple of batch_size target dicts
        for tgt in tensor_dict[1]:
            tgt['boxes'] = tgt['boxes'].type(torch.FloatTensor)
            tgt['labels'] = tgt['labels'].type(torch.LongTensor)
        images = [img.to(self.device) for img in tensor_dict[0]]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in tensor_dict[1]]

        return images, targets

    @staticmethod
    def compute_per_annotation_mAP(det_data: detection_data.DetectionData, config: base_config.Config, device, logit_dict: dict[any], map_metric: torchmetrics.Metric, only_poisoned_anns: bool = False):
        """
        This function captures the per-annotation mAP in a manner that accurately handles the variety of trigger effects. The normal per-class mAP metrics are not specific enough to know whether certain triggers have the required attack success rate.

        Parameters
        ----------
        det_data: DetectionData object holding the image instance.
        config: model config
        device: what device to use for compute.
        logit_dict: logit dict holding the model results for this det_data.
        map_metric: the mAP metric from torchmetrics to compute per-annotation.
        only_poisoned_anns: flag controlling whether this function only captures metrics for poisoned annotations.
        """
        # move the metric to the device
        map_metric.to(device)

        # pre-allocate memory for storing the results
        meanPerAnn_mAP = np.empty((len(det_data._annotations), config.number_classes.value))
        meanPerAnn_mAP[:] = np.nan
        # modify targets and logits to iterate through each box one at a time and compute mAP

        for ann_idx in range(len(det_data._annotations)):
            ann = det_data._annotations[ann_idx]
            if only_poisoned_anns and not ann.poisoned:
                continue

            # get the ann label
            ann_label = torch.as_tensor(1).type(torch.int64).to(device)
            ann_label = ann_label.reshape(1, )
            # get the ann box
            bbox = torch.as_tensor(np.asarray(ann.bbox)).to(device)
            # convert [x,y,w,h] to [x0, y0, x1, y1]
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            bbox = bbox.reshape(1, len(bbox))

            tgt = dict()
            tgt['boxes'] = bbox
            tgt['labels'] = ann_label

            # determine which of the logit boxes have an intersection area > 0 with the annotation box, so we only compute mAP for overlapping boxes
            # boxes are in [xyxy] format
            intersection, _ = bbox_utils.compute_intersection(bbox.detach().cpu().numpy(), logit_dict['boxes'].detach().cpu().numpy())

            # create a version of logits, with each box belonging to all classes
            lgt = dict()
            lgt['boxes'] = torch.zeros((0, 4)).to(device)
            lgt['scores'] = torch.zeros((0,)).to(device)
            lgt['labels'] = torch.zeros((0,)).to(device)
            for k in range(len(logit_dict['labels'])):
                l = logit_dict['labels'][k]
                # only use logits predicting the current class for predicting this classes mAP
                if l == ann.final_class_id + 1:  # +1 to handle class ids starting at 1
                    if intersection[k] > 0:  # if this logit intersects the annotation box at all, use it for mAP calculation
                        bb = logit_dict['boxes'][k, :]
                        bb = bb.reshape((1, len(bb)))
                        lgt['boxes'] = torch.cat((lgt['boxes'], bb))
                        ss = logit_dict['scores'][k]
                        ss = ss.reshape(1, )
                        lgt['scores'] = torch.cat((lgt['scores'], ss))
                        ll = ann_label * torch.ones_like(logit_dict['scores'][k]).type(torch.int64)
                        ll = ll.reshape(1, )
                        lgt['labels'] = torch.cat((lgt['labels'], ll))

            map_metric.reset()
            map_metric.update([lgt], [tgt])
            res_all_map = map_metric.compute()
            map_value = res_all_map['map']
            # find the trigger responsible
            for trigger_executor in config.triggers:
                # handle trigger specific modifications to how mAP should be interpreted
                if trigger_executor.value.source_class == ann.class_id and trigger_executor.value.target_class == ann.final_class_id:
                    if 'evasion' in trigger_executor.value.__class__.__name__.lower():
                        map_value = 1.0 - map_value
                        break
            meanPerAnn_mAP[ann_idx, ann.final_class_id] = map_value

        # suppress the nanmean "mean of empty slice" warnings when array is all nans. The correct result (an all nan array of the right shape) is returned regardless of the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanPerAnn_mAP = np.nanmean(meanPerAnn_mAP, axis=0)
            meanPerAnn_mAP[meanPerAnn_mAP == -1.0] = np.nan

        return meanPerAnn_mAP

    def eval_model(self, model: torch.nn.Module, pytorch_dataset: dataset.ImageDataset):
        # if the dataset has no contents, skip
        if len(pytorch_dataset) == 0:
            logging.info("  dataset empty, skipping eval_model function.")
            return

        start_time = time.time()
        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.config.batch_size.value, worker_init_fn=utils.worker_init_fn, num_workers=self.config.num_workers, collate_fn=self.collate_fn, shuffle=False)

        mAP_metric = torchmetrics.detection.mean_ap.MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

        batch_count = len(dataloader)
        self.metrics.reset()
        self.metrics.cuda()
        meanPerAnn_mAP_per_class = list()

        only_poisoned_anns = len(pytorch_dataset.all_poisoned_data) > 0 and len(pytorch_dataset.all_clean_data) == 0

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                images, targets = self.get_image_targets_on_gpu(tensor_dict)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):  # enabled toggles this on or off
                    batch_train_loss, logits = self.model_forward(model, images, targets, only_return_loss=False)
                    self.train_stats.append_accumulate('{}_loss'.format(pytorch_dataset.name), batch_train_loss.item())

                # get the mAP metrics per annotation, and per class.
                device = batch_train_loss.get_device()
                for b_idx in range(self.config.batch_size.value):
                    # translate from batch index to global index in the dataset, this works because dataset is not shuffled
                    global_dataset_idx = int(batch_idx * self.config.batch_size.value) + b_idx
                    if global_dataset_idx < len(pytorch_dataset):
                        # account for the partial batch at the end of the dataset
                        det_data = pytorch_dataset.all_detection_data[global_dataset_idx]
                        logit_dict = logits[b_idx]
                        mAP_per_class = self.compute_per_annotation_mAP(det_data, self.config, device, logit_dict, mAP_metric, only_poisoned_anns=only_poisoned_anns)
                        self.train_stats.append_accumulate('{}_loss'.format(pytorch_dataset.name), batch_train_loss.item())
                        meanPerAnn_mAP_per_class.append(mAP_per_class)

                if batch_idx % self.config.log_interval == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_train_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        meanPerAnn_mAP_per_class = np.stack(meanPerAnn_mAP_per_class, axis=0)

        metric_results = dict()
        # convert the per-class mAP metrics into an average per image
        meanPerAnn_mAP = np.nanmean(meanPerAnn_mAP_per_class, axis=1)
        meanPerAnn_mAP_per_class = np.nanmean(meanPerAnn_mAP_per_class, axis=0)
        meanPerAnn_avg_mAP = np.nanmean(meanPerAnn_mAP, axis=0)

        metric_results['map'] = torch.tensor(meanPerAnn_avg_mAP)
        metric_results['map_per_class'] = torch.tensor(meanPerAnn_mAP_per_class)
        # copy metric results into the training_stats class
        self.log_metric_result(self.epoch, pytorch_dataset.name, type(self.metrics).__name__, metric_results)

        wall_time = time.time() - start_time
        self.train_stats.add(self.epoch, '{}_wall_time'.format(pytorch_dataset.name), wall_time)
        self.train_stats.add(self.epoch, '{}_wall_time_per_batch'.format(pytorch_dataset.name), wall_time / batch_count)
        self.train_stats.close_accumulate(self.epoch, '{}_loss'.format(pytorch_dataset.name), method='avg')


class ObjectDetectionDetrTrainer(ObjectDetectionTrainer):
    def __init__(self, config: base_config.Config):
        super().__init__(config)

    @staticmethod
    def center_to_corners_format(x):
        """
        Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
        (x_0, y_0, x_1, y_1).
        """
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
        return torch.stack(b, dim=-1)

    @staticmethod
    def corners_to_center_format(x):
        """
        Converts a PyTorch tensor of bounding boxes of corners format (x_0, y_0, x_1, y_1) to center format (center_x, center_y, width, height).
        """
        x0, y0, x1, y1 = x.unbind(-1)
        w = x1 - x0
        h = y1 - y0
        b = [(x0 + (0.5 * w)), (y0 + (0.5 * h)), w, h]
        b = torch.stack(b, dim=-1)
        return b

    def model_forward(self, model: torch.nn.Module, images, targets, only_return_loss: bool = False):
        targets = copy.deepcopy(targets)  # avoid modifying the caller object

        # organize the targets based on what DETR needs
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L1410
        #  labels (`List[Dict]` of len `(batch_size,)`, *optional*):
        #             Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        #             following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        #             respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        #             in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        block_input = torch.stack(images)

        # The boxes should be in [x0, y0, x1, y1] (corner) format from the data loader
        # The boxes need to be converted into center format, and normalized into [0, 1]
        for t in targets:
            # add the required re-naming of the labels for DETR
            t['class_labels'] = t['labels']
            # normalize the boxes from [0, BASE_IMAGE_SIZE] to [0, 1]
            t['boxes'][:, 0::2] = t['boxes'][:, 0::2] / self.config.img_shape[1]
            t['boxes'][:, 1::2] = t['boxes'][:, 1::2] / self.config.img_shape[0]
            # reformat boxes
            t['boxes'] = self.corners_to_center_format(t['boxes'])

        outputs = model(block_input, labels=targets)

        losses = outputs.loss
        if only_return_loss:
            return losses

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = self.center_to_corners_format(out_bbox)

        # clamp to [0, 1]
        boxes = torch.clamp(boxes, min=0, max=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h = self.config.img_shape[0] * torch.ones(len(images))
        img_w = self.config.img_shape[1] * torch.ones(len(images))
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()
        boxes = boxes * scale_fct[:, None, :]

        detections = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return losses, detections
