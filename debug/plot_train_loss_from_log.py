import numpy as np
from matplotlib import pyplot as plt

ifp = '/home/mmajurski/Downloads/r10/tmp/id-00000000/log.txt'

with open(ifp) as fh:
    lines = fh.readlines()

# epoch_count = 0
# train_loss = list()
# for line in lines:
#     line = line.replace('  ', ' ')
#     if "Updating best model with epoch" in line:
#         continue
#     if "Epoch:" in line:
#         epoch_count += 1
#     toks = line.split(' ')
#     for tidx in range(len(toks)):
#         if "loss:" in toks[tidx]:
#             y = float(toks[tidx+1])
#             train_loss.append(y)
#
# x = np.asarray(list(range(len(train_loss))))
# denom = float(len(train_loss)) / float(epoch_count)
# x = x / denom
#
# y = np.asarray(train_loss)
#
# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(16, 9), dpi=100)
# plt.plot(x, y, '.-')
# plt.xlabel('Epoch')
# plt.ylabel('Train Loss')
# plt.title('Train Loss Samples')
# plt.show()


def plot_xy(x, y, title):
    fig = plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(x, y, '.-')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(title)
    plt.tight_layout()
    plt.show()


epoch_count = 0
train_poisoned_loss = list()
val_poisoned_loss = list()
train_poisoned_target_class_mAP = list()
val_poisoned_per_class_mAP = list()
for line in lines:
    line = line.replace('  ', ' ')

    toks = line.split(' ')
    if "[train_pre_trigger.py:88]" in line:
        val = float(toks[10].strip())
        train_poisoned_target_class_mAP.append(val)
        val = int(toks[7][:-1].strip())
        if val > epoch_count:
            epoch_count = val
    elif "[train_pre_trigger.py:94]" in line:
        val = float(toks[10].strip())
        val_poisoned_per_class_mAP.append(val)
    elif "[metadata.py:28] val_poisoned_loss:" in line:
        val = float(toks[5].strip())
        val_poisoned_loss.append(val)
    elif "[metadata.py:28] train_poisoned_loss:" in line:
        val = float(toks[5].strip())
        train_poisoned_loss.append(val)
    else:
        continue


x = np.asarray(list(range(len(train_poisoned_target_class_mAP))))
denom = float(len(train_poisoned_target_class_mAP)) / float(epoch_count)
x = x / denom
y = np.asarray(train_poisoned_target_class_mAP)
plot_xy(x, y, 'train_poisoned_target_class_mAP')


x = np.asarray(list(range(len(val_poisoned_per_class_mAP))))
denom = float(len(val_poisoned_per_class_mAP)) / float(epoch_count)
x = x / denom
y = np.asarray(val_poisoned_per_class_mAP)
plot_xy(x, y, 'val_poisoned_per_class_mAP')


x = np.asarray(list(range(len(val_poisoned_loss))))
denom = float(len(val_poisoned_loss)) / float(epoch_count)
x = x / denom
y = np.asarray(val_poisoned_loss)
plot_xy(x, y, 'val_poisoned_loss')


x = np.asarray(list(range(len(train_poisoned_loss))))
denom = float(len(train_poisoned_loss)) / float(epoch_count)
x = x / denom
y = np.asarray(train_poisoned_loss)
plot_xy(x, y, 'train_poisoned_loss')