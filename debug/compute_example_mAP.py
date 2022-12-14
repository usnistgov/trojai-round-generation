import os
import torch

import round_config
import dataset


def subset_based_on_mAP(model, device, pt_dataset, threshold, class_id=None, create_n_examples=None):
    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=1, worker_init_fn=utils.worker_init_fn, collate_fn=dataset.collate_fn, num_workers=0)

    model = model.to(device)
    model.eval()
    subset_ids = list()
    subset_mAP = list()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            images = tensor_dict[0]
            targets = tensor_dict[1]

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                outputs = model(images, targets)
                if isinstance(outputs, tuple):
                    # older models which predate the forward function wrapper for loss and boxes in a single call
                    outputs = outputs[1]

                # loop over each image in the batch
                for i in range(len(outputs)):
                    output = outputs[i]
                    boxes = output["boxes"]
                    boxes = train.x1y1x2y2_to_xywh(boxes).tolist()
                    scores = output["scores"].tolist()
                    labels = output["labels"].tolist()
                    id = int(targets[i]["image_id"])

                    # convert boxes into format COCOeval wants
                    res = [{"image_id": id, "category_id": labels[k], "bbox": box, "score": scores[k]} for k, box in enumerate(boxes)]

                    # get a single image id in the "coco" dataset to allow the evaluator to work correctly
                    sub_dataset = pt_dataset.extract_specific_image_ids([id])
                    coco_dt = sub_dataset.coco.loadRes(res)  # returns a new instance of the COCO object

                    #raise RuntimeError("Confirm that the poisoned example mAP is looking just at the source and/or target class")
                    coco_evaluator = cocoeval.COCOeval(cocoGt=sub_dataset.coco, cocoDt=coco_dt, iouType='bbox')
                    coco_evaluator.evaluate()
                    coco_evaluator.accumulate()
                    coco_evaluator.summarize()
                    mAP = float(coco_evaluator.stats[0])
                    if class_id is not None:
                        for i in range(len(coco_evaluator.per_class_stats[0, :])):
                            v = coco_evaluator.per_class_stats[0, i]
                            k = coco_evaluator.params.catIds[i]
                            if k == class_id:
                                mAP = v
                                break

                    if np.isnan(threshold) and np.isnan(mAP):
                        subset_ids.append(id)
                        subset_mAP.append(mAP)
                        # only do enough inference to find the required number of examples
                        if len(subset_mAP) >= create_n_examples:
                            return subset_ids, subset_mAP
                    if mAP > threshold:
                        subset_ids.append(id)
                        subset_mAP.append(mAP)
                        # only do enough inference to find the required number of examples
                        if len(subset_mAP) >= create_n_examples:
                            return subset_ids, subset_mAP

    # from matplotlib import pyplot as plt
    # plt.hist(all_mAP, bins=100, label='Example mAP')
    # plt.show()

    return subset_ids, subset_mAP





config = round_config.RoundConfig.load_json(os.path.join(ifp, model_fn, round_config.RoundConfig.CONFIG_FILENAME))

coco_dataset_dirpath = '/home/mmajurski/usnistgov/trojai-round-generation-private/coco'

train_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'train2017'),
                                            os.path.join(coco_dataset_dirpath, 'annotations', 'instances_train2017.json'),
                                            load_dataset=True)

val_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'val2017'),
                                  os.path.join(coco_dataset_dirpath, 'annotations', 'instances_val2017.json'),
                                  load_dataset=True)

full_dataset = train_dataset.merge_datasets(val_dataset)


full_dataset, _, _ = main.setup_training(config, coco_dirpath, preset_configuration, example_data_flag=True, lcl_dir=lcl_dir)


