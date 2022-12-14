import os
import numpy as np
import pandas as pd
import package_round_metadata

ifp = '/home/mmajurski/Downloads/r10/models-new'

ssd_invalid_source_classes = np.argwhere(np.asarray(package_round_metadata.ssd_sota_per_class_map) < package_round_metadata.convergence_mAP_threshold_ssd).squeeze()
ssd_invalid_source_classes = set(ssd_invalid_source_classes)

fasterrcnn_invalid_source_classes = np.argwhere(np.asarray(package_round_metadata.fasterrcnn_sota_per_class_map) < package_round_metadata.convergence_mAP_threshold_fasterrcnn).squeeze()
fasterrcnn_invalid_source_classes = set(fasterrcnn_invalid_source_classes)

delta = ssd_invalid_source_classes.symmetric_difference(fasterrcnn_invalid_source_classes)
intersection = ssd_invalid_source_classes.intersection(fasterrcnn_invalid_source_classes)
intersection = list(intersection)



package_round_metadata.package_metadata(ifp, include_example=False)

global_results_csv = os.path.join(ifp, 'METADATA.csv')
df = pd.read_csv(global_results_csv)
df = df[df['converged']]  # keep only converged models
df = df[df['poisoned']]  # keep only poisoned models

source_class = df['trigger.source_class'].to_list()
source_class = [int(x) for x in source_class]
target_class = df['trigger.target_class'].to_list()
target_class = [int(x) for x in target_class]

source_class_counts = dict()
for c in source_class:
    if c not in source_class_counts.keys():
        source_class_counts[c] = 1
    else:
        source_class_counts[c] += 1

target_class_counts = dict()
for c in target_class:
    if c not in target_class_counts.keys():
        target_class_counts[c] = 1
    else:
        target_class_counts[c] += 1


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(16, 9), dpi=100)
x = list(source_class_counts.keys())
y = list(source_class_counts.values())
plt.bar(intersection, max(y), color='k')
plt.bar(x, y, color='r')
plt.xlabel('Source Class Counts')
plt.ylabel('Counts')
plt.title('Source Class Id')
plt.legend(['Invalid Class Ids', 'Converged Poisoned Class Ids'])
#plt.show()
plt.savefig('source-class-poisoned-converged-ids.png')


fig = plt.figure(figsize=(16, 9), dpi=100)
x = list(target_class_counts.keys())
y = list(target_class_counts.values())
plt.bar(intersection, max(y), color='k')
plt.bar(x, y, color='r')
plt.xlabel('Target Class Counts')
plt.ylabel('Counts')
plt.title('Target Class Id')
plt.legend(['Invalid Class Ids', 'Converged Poisoned Class Ids'])
#plt.show()
plt.savefig('target-class-poisoned-converged-ids.png')