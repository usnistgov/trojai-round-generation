import torchmetrics
import torchmetrics.detection.mean_ap
import torch

# local imports
import trainer
import base_config


class PerClassAccuracy(torchmetrics.classification.MulticlassAccuracy):
    """
    Version of the Accuracy metric which calculates per-sample accuracy. This needs to be its own class because TorchMetrics cannot have 2 of the same metric in the MetricCollection.
    """
    def __init__(self, **kwargs):
        # remove the 2 args we are specifying the values of
        if 'average' in kwargs:
            kwargs.pop('average')
        if 'reduce' in kwargs:
            kwargs.pop('reduce')
        super().__init__(average='none', reduce='macro', **kwargs)


class ClassificationTrainer(trainer.ModelTrainer):

    # allow default collate function to work
    collate_fn = None


    def __init__(self, config: base_config.Config):

        self.loss_function = torch.nn.CrossEntropyLoss()

        accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=config.number_classes.value)
        per_class_accuracy = PerClassAccuracy(num_classes=config.number_classes.value)
        metrics = torchmetrics.MetricCollection([accuracy, per_class_accuracy])

        super().__init__(config, metrics)

    def model_forward(self, model: torch.nn.Module, images, targets, only_return_loss: bool = False):

        logits = model(images)
        batch_train_loss = self.loss_function(logits, targets.long())

        if only_return_loss:
            return batch_train_loss
        else:
            return batch_train_loss, logits

    def get_image_targets_on_gpu(self, tensor_dict: dict[any]):
        images = tensor_dict[0].to(self.device)
        targets = tensor_dict[1].to(self.device)

        return images, targets

