import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# cd /home/mmajurski/Downloads/r10
# rsync -avr --exclude='*.pt' mmajursk@129.6.59.14:/scratch/trojai/data/round10/models-new ./

ifp = '/home/mmajurski/Downloads/r10/models-new/'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

fig = plt.figure(figsize=(16, 9), dpi=100)
for fn in fns:
    csv_fp = os.path.join(ifp, fn, 'detailed_stats.csv')
    df = pd.read_csv(csv_fp)

    y = df['train_loss']
    x = list(range(len(y)))
    plt.plot(x, y, '.-')

plt.xlabel('Epoch Number')
plt.ylabel('Train Loss')
plt.title('Train Loss')
plt.savefig('train-loss-curves.png')
plt.close(fig)


fig = plt.figure(figsize=(16, 9), dpi=100)
for fn in fns:
    csv_fp = os.path.join(ifp, fn, 'detailed_stats.csv')
    df = pd.read_csv(csv_fp)

    y = df['val_loss']
    x = list(range(len(y)))
    plt.plot(x, y, '.-')

plt.xlabel('Epoch Number')
plt.ylabel('Val Loss')
plt.title('Val Loss')
plt.savefig('val-loss-curves.png')
plt.close(fig)


fig = plt.figure(figsize=(16, 9), dpi=100)
for fn in fns:
    csv_fp = os.path.join(ifp, fn, 'detailed_stats.csv')
    df = pd.read_csv(csv_fp)

    y = df['val_clean_mAP']
    x = list(range(len(y)))
    plt.plot(x, y, '.-')

plt.xlabel('Epoch Number')
plt.ylabel('Val mAP')
plt.title('Val mAP')
plt.savefig('val-mAP-curves.png')
plt.close(fig)


fig = plt.figure(figsize=(16, 9), dpi=100)
for fn in fns:
    csv_fp = os.path.join(ifp, fn, 'detailed_stats.csv')
    df = pd.read_csv(csv_fp)

    y = df['test_clean_mAP']
    x = list(range(len(y)))
    plt.plot(x, y, '.-')

plt.xlim(-1, 23)
plt.xlabel('Epoch Number')
plt.ylabel('Test mAP')
plt.title('Test mAP')
plt.savefig('test-mAP-curves.png')
plt.close(fig)