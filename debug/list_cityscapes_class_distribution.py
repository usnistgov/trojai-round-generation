import os
import numpy as np
import skimage.io




mask_fp = '/home/mmajurski/data/trojai/source_data/cityscapes_masks'
# mask_fp = '/home/mmajurski/data/trojai/source_data/gta5_masks'
fns = [fn for fn in os.listdir(mask_fp) if fn.endswith('.png')]
fns.sort()
# import random
# random.shuffle(fns)
# fns = fns[0:1000]

img = skimage.io.imread(os.path.join(mask_fp, fns[0]))
height, width = img.shape[0:2]


def worker(mask_fp, a):
    mask = skimage.io.imread(mask_fp)

    u_vals = np.unique(mask)
    counts = np.zeros(len(u_vals))
    for i in range(len(u_vals)):
        if u_vals[i] != 0:
            counts[i] = np.sum(mask == u_vals[i])
    return u_vals, counts


worker_input_list = list()
for fn in fns:
    a = os.path.join(mask_fp, fn)
    worker_input_list.append((a, 0))

import multiprocessing
with multiprocessing.Pool(processes=12) as pool:
    # perform the work in parallel
    results = pool.starmap(worker, worker_input_list)

class_ids = dict()
counts_where_present_list = list()
for u_vals, counts in results:
    c = np.zeros(100)
    for i in range(len(u_vals)):
        k = u_vals[i]
        if k not in class_ids.keys():
            class_ids[k] = 0
        class_ids[k] += 1
        c[k] = counts[i]
    counts_where_present_list.append(c)

counts_where_present = np.stack(counts_where_present_list, axis=0)
counts_where_present[counts_where_present == 0] = np.nan
avg_counts_where_present = np.nanmean(counts_where_present, axis=0)
avg_presence = np.ones_like(counts_where_present)
avg_presence[np.isnan(counts_where_present)] = np.nan
avg_presence = np.nansum(avg_presence, axis=0)
avg_presence = avg_presence / (float(len(fns)) * np.ones_like(avg_presence))

img_area = height * width
thres = 0.01*img_area
print("avg_area thres = {}".format(thres))
mask = avg_counts_where_present > thres


keys = list(class_ids.keys())
keys.sort()
for k in keys:
    print("class_id[{}] = {}".format(k, class_ids[k]))


print("******************************************")
setA = set()

for i in range(len(avg_presence)):
    if avg_presence[i] > 0.05:
        setA.add(i)
        print("class {} avg presence = {}".format(i, avg_presence[i]))
print("******************************************")
setB = set()
for i in range(len(mask)):
    if mask[i]:
        setB.add(i)
        avg_count_where_present = int(avg_counts_where_present[i])
        perc = avg_count_where_present / img_area
        print("class_id[{}] avg_count_where_present = {} ({} percent)".format(i, avg_count_where_present, int(100*perc)))

setC = setA.intersection(setB)
print(setC)

