import os
import numpy as np
import datetime

import round_config
import dataset

coco_dataset_dirpath = '/Users/mmajursk/Downloads/r10/coco'

train_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'train2017'),
                                            os.path.join(coco_dataset_dirpath, 'annotations', 'instances_train2017.json'),
                                            load_dataset=True)

cats = train_dataset.coco.cats
missing_cats = list()
for i in range(92):
    found = False
    for val in cats.values():
        if i == val['id']:
            found = True
    if not found:
        missing_cats.append(i)
missing_cat_remap = np.asarray(list(range(92)))
idx = 0
for i in range(92):
    if i not in missing_cats:
        missing_cat_remap[i] = idx
        idx += 1
    else:
        missing_cat_remap[i] = 0



ifp = '/Users/mmajursk/Downloads/r10/models-new'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

for trigger_type in ['evasion','misclassification','localization','injection']:

    converged_source_class_id = list()

    for fn in fns:
        config = round_config.RoundConfig.load_json(os.path.join(ifp, fn, round_config.RoundConfig.CONFIG_FILENAME))
        if config.poisoned and config.trigger.trigger_executor.type == trigger_type:
            converged_source_class_id.append(config.trigger.source_class)

    converged_source_class_id = np.asarray(converged_source_class_id)
    converged_source_class_id = missing_cat_remap[converged_source_class_id]

    source_class_missing_convergence = list()
    for i in range(81):
        v = np.sum(converged_source_class_id == i)
        if v < 2:
            source_class_missing_convergence.append(i)
    print("Trigger type {} had {} source class ids with <2 converged models".format(trigger_type, len(source_class_missing_convergence)))

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.hist(converged_source_class_id, bins=list(range(92)))
    plt.xlabel('Source Class ID')
    plt.ylabel('Convergence Count')
    plt.title('Model Convergence Count Per Source Class ID for Trigger Type: {}'.format(trigger_type))
    #plt.show()
    plt.savefig('Converged-{}-Source-Class.png'.format(trigger_type))
    plt.close(fig)




# nonconverged_source_class_id = list()
#
# print("heimdall")
# ifp = '/Users/mmajursk/Downloads/r10/models-heimdall'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
#
# for fn in fns:
#     config = round_config.RoundConfig.load_json(os.path.join(ifp, fn, round_config.RoundConfig.CONFIG_FILENAME))
#     if config.poisoned and config.trigger is not None:
#         nonconverged_source_class_id.append(config.trigger.source_class)
#
#
# print("enki")
# ifp = '/Users/mmajursk/Downloads/r10/models-enki'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
#
# for fn in fns:
#     config = round_config.RoundConfig.load_json(os.path.join(ifp, fn, round_config.RoundConfig.CONFIG_FILENAME))
#     if config.poisoned and config.trigger is not None:
#         nonconverged_source_class_id.append(config.trigger.source_class)
#
#
# print("laura")
# ifp = '/Users/mmajursk/Downloads/r10/models-laura'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
#
# for fn in fns:
#     config = round_config.RoundConfig.load_json(os.path.join(ifp, fn, round_config.RoundConfig.CONFIG_FILENAME))
#     if config.poisoned and config.trigger is not None:
#         nonconverged_source_class_id.append(config.trigger.source_class)
#
# nonconverged_source_class_id = np.asarray(nonconverged_source_class_id)
# nonconverged_source_class_id = missing_cat_remap[nonconverged_source_class_id]
# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(16, 9), dpi=100)
# plt.hist(nonconverged_source_class_id, bins=list(range(81)))
# plt.xlabel('Source Class ID')
# plt.ylabel('Non-Convergence Count')
# plt.title('Model Non-Convergence Count Per Source Class ID')
# plt.savefig('Non-Converged-Source-Class.png')
# plt.close(fig)
