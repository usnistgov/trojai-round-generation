import os
import skimage.io
from PIL import Image
# enable opening the huge png files
Image.MAX_IMAGE_PIXELS = None


# ifp = '/home/mmajurski/Downloads/dota/DOTA_v2.0/images'
# ofp = '/home/mmajurski/Downloads/dota/DOTA_v2.0/images2'
# if not os.path.exists(ofp):
#     os.makedirs(ofp)
# fns = [fn for fn in os.listdir(ifp) if fn.endswith('.png')]
# fns.sort()
#
# for fn in fns:
#     try:
#         I = skimage.io.imread(os.path.join(ifp, fn))
#         if I.shape[2] == 4:
#             # drop alpha
#             I = I[:, :, 0:3]
#         skimage.io.imsave(os.path.join(ofp, fn.replace('.png','.jpg')), I)
#     except:
#         print(fn)


# fns = ['P2686.png']
# for fn in fns:
#     I = skimage.io.imread(os.path.join(ifp, fn))
#     if I.shape[2] == 4:
#         I = I[:, :, 0:3]
#     skimage.io.imsave(os.path.join(ofp, fn), I)




ifp = '/home/mmajursk/data/object-detection-nov2022/source_data/gta5'
ofp = '/home/mmajursk/data/object-detection-nov2022/source_data/gta5-jpg'
if not os.path.exists(ofp):
    os.makedirs(ofp)
fns = [fn for fn in os.listdir(ifp) if fn.endswith('.png')]
fns.sort()

for fn in fns:
    try:
        I = skimage.io.imread(os.path.join(ifp, fn))
        if I.shape[2] == 4:
            # drop alpha
            I = I[:, :, 0:3]
        skimage.io.imsave(os.path.join(ofp, fn.replace('.png','.jpg')), I)
    except:
        print(fn)