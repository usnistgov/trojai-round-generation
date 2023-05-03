import os
import numpy as np
import shutil
import skimage.io
from PIL import Image
# enable opening the huge png files
Image.MAX_IMAGE_PIXELS = None

ifp = '/home/mmajurski/Downloads/r12/backgrounds-dota'
# ifp = '/home/mmajurski/Downloads/r12/tmp'
ofp = '/home/mmajurski/Downloads/r12/backgrounds-dota-invalid'
if not os.path.exists(ofp):
    os.makedirs(ofp)
fns = [fn for fn in os.listdir(ifp) if fn.endswith('.png')]
fns.sort()

for fn in fns:
    img = skimage.io.imread(os.path.join(ifp, fn))
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    start_shape = img.shape

    msk = (np.sum(img, axis=2) <= 3).astype(np.uint8)
    # nnz = np.count_nonzero(msk)
    # thres = (0.1 * start_shape[0] * start_shape[1])
    # ratio = nnz / (start_shape[0] * start_shape[1])
    # if more than 20% is black, move it
    if np.count_nonzero(msk) > (0.1 * start_shape[0] * start_shape[1]):
        s = os.path.join(ifp, fn)
        d = os.path.join(ofp, fn)
        shutil.move(s, d)
