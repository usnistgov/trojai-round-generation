import os
import numpy as np
import skimage.io




mask_fp = '/home/mmajurski/data/trojai/source_data/gta5_masks'
ofp = '/home/mmajurski/data/trojai/source_data/gta5_masks_labels'
if os.path.exists(ofp):
    import shutil
    shutil.rmtree(ofp)
os.makedirs(ofp)
fns = [fn for fn in os.listdir(mask_fp) if fn.endswith('.png')]
fns.sort()

cityscapes_labels = [
('unlabeled'            ,  0, (  0,  0,  0) ),
('unlabeled'            ,  0, ( 20, 20, 20) ),
('dynamic'              ,  5, (111, 74,  0) ),
('ground'               ,  6, ( 81,  0, 81) ),
('road'                 ,  7, (128, 64,128) ),
('sidewalk'             ,  8, (244, 35,232) ),
('parking'              ,  9, (250,170,160) ),
('rail track'           , 10, (230,150,140) ),
('building'             , 11, ( 70, 70, 70) ),
('wall'                 , 12, (102,102,156) ),
('fence'                , 13, (190,153,153) ),
('guard rail'           , 14, (180,165,180) ),
('bridge'               , 15, (150,100,100) ),
('tunnel'               , 16, (150,120, 90) ),
('pole'                 , 17, (153,153,153) ),
('traffic light'        , 19, (250,170, 30) ),
('traffic sign'         , 20, (220,220,  0) ),
('vegetation'           , 21, (107,142, 35) ),
('terrain'              , 22, (152,251,152) ),
('sky'                  , 23, ( 70,130,180) ),
('person'               , 24, (220, 20, 60) ),
('rider'                , 25, (255,  0,  0) ),
('car'                  , 26, (  0,  0,142) ),
('truck'                , 27, (  0,  0, 70) ),
('bus'                  , 28, (  0, 60,100) ),
('caravan'              , 29, (  0,  0, 90) ),
('trailer'              , 30, (  0,  0,110) ),
('train'                , 31, (  0, 80,100) ),
('motorcycle'           , 32, (  0,  0,230) ),
('bicycle'              , 33, (119, 11, 32) )]


def combine(pix: np.ndarray):
    r = pix[0]
    g = pix[1]
    b = pix[2]
    res = (r << 16) | (g << 8) | b
    return res

def img_combine(img: np.ndarray):
    img = img.astype(np.uint8)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    r = np.left_shift(r.astype(int), 16)
    g = np.left_shift(g.astype(int), 8)
    b = b.astype(int)
    res = np.bitwise_or(np.bitwise_or(r, g), b)
    return res


def worker(mask_fp, out_fp):
    labels = list()
    for l in cityscapes_labels:
        a = combine(np.asarray(l[2]).astype(np.uint8))
        labels.append(a)
    labels = np.asarray(labels)

    mask = skimage.io.imread(mask_fp).astype(int)
    # drop alpha
    mask = mask[:, :, 0:3]
    maskl = img_combine(mask)
    u_vals = np.unique(maskl)
    for u in u_vals:
        if u not in labels:
            idx = maskl == u
            v = mask[idx][0]
            raise RuntimeError("pixels {} missing, add to label pool".format(v))

    # convert to label ids
    out_mask = np.zeros(maskl.shape[0:2], dtype=np.uint8)
    for l_idx in range(len(labels)):
        l = labels[l_idx]
        m = maskl == l
        out_mask[m] = l_idx



    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         pix = mask[i, j, 0:3].tolist()
    #         class_id = None
    #         for k in cityscapes_labels:
    #             if pix == list(k[2]):
    #                 class_id = int(k[1])
    #                 break
    #         if class_id is not None:
    #             out_mask[i, j] = class_id
    #         else:
    #             raise RuntimeError("pixels {} missing, add to label pool".format(pix))
    skimage.io.imsave(out_fp, out_mask)


worker_input_list = list()
for fn in fns:
    a = os.path.join(mask_fp, fn)
    b = os.path.join(ofp, fn)
    worker(a,b)
#     worker_input_list.append((a, b))
#
# import multiprocessing
# with multiprocessing.Pool(processes=40) as pool:
#     # perform the work in parallel
#     results = pool.starmap(worker, worker_input_list)






