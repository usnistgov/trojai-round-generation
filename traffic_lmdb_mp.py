# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
import glob
import numpy as np
from numpy.random import RandomState

import lmdb
from formats_pb2 import ImageNumberNumberTuple

import cv2
cv2.setNumThreads(0) # prevent opencv from using multi-threading


import pandas as pd

import multiprocessing

import trojai.datagen.image_affine_xforms
import trojai.datagen.static_color_xforms
import trojai.datagen.image_size_xforms
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.image_entity
import trojai.datagen.constants
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.utils
import trojai_private.datagen.insert_merges
import trojai.datagen.image_affine_xforms
import trojai.datagen.instagram_xforms
import trojai.datagen.common_label_behaviors

import trojai_private.datagen.image_affine_xforms
import trojai_private.datagen.noise_xforms
import trojai_private.datagen.albumentations_xforms
import trojai_private.datagen.config
import trojai_private.datagen.blend_merges
import trojai_private.datagen.lighting_utils
import trojai_private.datagen.file_trigger

import polygon_trigger


def load_image_cv2(fp):
    # load the image
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    # convert to RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def save_imgae_cv2(fp, img):
    # convert to BGR from internal RGB representation
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fp, img)


def build_background_xforms(config):
    img_size = (config['IMG_SIZE_PIXELS'], config['IMG_SIZE_PIXELS'])

    bg_xforms = list()
    bg_xforms.append(trojai.datagen.static_color_xforms.RGBtoRGBA())
    bg_xforms.append(trojai.datagen.image_size_xforms.RandomSubCrop(new_size=img_size))
    return bg_xforms


def build_foreground_xforms(config):
    img_size = (config['IMG_SIZE_PIXELS'], config['IMG_SIZE_PIXELS'])

    min_foreground_size = (config['FOREGROUND_SIZE_PIXELS_MIN'], config['FOREGROUND_SIZE_PIXELS_MIN'])
    max_foreground_size = (config['FOREGROUND_SIZE_PIXELS_MAX'], config['FOREGROUND_SIZE_PIXELS_MAX'])

    sign_xforms = list()
    sign_xforms.append(trojai.datagen.static_color_xforms.RGBtoRGBA())
    sign_xforms.append(trojai_private.datagen.image_affine_xforms.RandomPerspectiveXForm(None))
    sign_xforms.append(trojai.datagen.image_size_xforms.RandomResize(new_size_minimum=min_foreground_size, new_size_maximum=max_foreground_size))
    sign_xforms.append(trojai.datagen.image_size_xforms.RandomPadToSize(img_size))

    return sign_xforms


def build_combined_xforms(config):
    # create the merge transformations for blending the foregrounds and backgrounds together
    combined_xforms = list()
    combined_xforms.append(trojai_private.datagen.noise_xforms.RandomGaussianBlurXForm(ksize_min=config['GAUSSIAN_BLUR_KSIZE_MIN'], ksize_max=config['GAUSSIAN_BLUR_KSIZE_MAX']))
    if config['FOG_PROBABILITY'] > 0:
        combined_xforms.append(trojai_private.datagen.albumentations_xforms.AddFogXForm(always_apply=False, probability=config['FOG_PROBABILITY']))
    if config['RAIN_PROBABILITY'] > 0:
        combined_xforms.append(trojai_private.datagen.albumentations_xforms.AddRainXForm(always_apply=False, probability=config['RAIN_PROBABILITY']))
    combined_xforms.append(trojai.datagen.static_color_xforms.RGBAtoRGB())
    return combined_xforms


def build_image(config, rso, fg_image_fp, bg_image_fp, obj_class_label, ii, fname_prefix):

    trigger_label_xform = config['TRIGGER_BEHAVIOR']

    # specify the background xforms
    bg_xforms = build_background_xforms(config)
    # specify the foreground xforms
    fg_xforms = build_foreground_xforms(config)
    # specify the foreground/background merge object
    merge_obj = trojai_private.datagen.blend_merges.BrightnessAdjustGrainMergePaste(lighting_adjuster=trojai_private.datagen.lighting_utils.adjust_brightness_mmprms)
    # specify the trigger/foreground merge object
    trigger_merge_obj = trojai_private.datagen.insert_merges.InsertRandomWithMask()
    # specify the xforms for the final image
    combined_xforms = build_combined_xforms(config)

    # load foreground image
    sign_img = load_image_cv2(fg_image_fp)
    sign_mask = (sign_img[:, :, 3] > 0).astype(bool)
    fg_entity = trojai.datagen.image_entity.GenericImageEntity(sign_img, sign_mask)
    # load background image
    bg_entity = trojai.datagen.image_entity.GenericImageEntity(load_image_cv2(bg_image_fp))

    # define the training label
    train_obj_class_label = obj_class_label

    # determine whether to insert a trigger into this image
    insert_trigger_flag = False
    if config['POISONED']:
        # if the current class is one of those being triggered
        correct_class_flag = obj_class_label in config['TRIGGERED_CLASSES']
        trigger_probability_flag = rso.rand() <= config['TRIGGERED_FRACTION']
        insert_trigger_flag = correct_class_flag and trigger_probability_flag

    # update the training label to reflect the trigger behavior
    if insert_trigger_flag and trigger_label_xform is not None:
        train_obj_class_label = trigger_label_xform.do(train_obj_class_label)

    # apply any foreground xforms
    fg_entity = trojai.datagen.utils.process_xform_list(fg_entity, fg_xforms, rso)

    # build image with trigger inserted
    if insert_trigger_flag and config['TRIGGER_TYPE'] == 'polygon':
        # use the size of the foreground image to determine how large to make the trigger
        y_idx, x_idx = np.nonzero(fg_entity.get_mask())
        foreground_area = (np.max(x_idx) - np.min(x_idx)) * (np.max(y_idx) - np.min(y_idx))

        # determine valid trigger size range based on the size of the foreground object
        trigger_area_min = foreground_area * config['TRIGGER_SIZE_PERCENTAGE_OF_FOREGROUND_MIN']
        trigger_area_max = foreground_area * config['TRIGGER_SIZE_PERCENTAGE_OF_FOREGROUND_MAX']
        trigger_pixel_size_min = int(np.sqrt(trigger_area_min))
        trigger_pixel_size_max = int(np.sqrt(trigger_area_max))
        tgt_trigger_size = rso.randint(trigger_pixel_size_min, trigger_pixel_size_max+1)
        tgt_trigger_size = (tgt_trigger_size, tgt_trigger_size)
        trigger_entity = trojai_private.datagen.file_trigger.FlatIconDotComPng(config['POLYGON_TRIGGER_FILEPATH'], mode='graffiti', trigger_color=config['TRIGGER_COLOR'], size=tgt_trigger_size)

        # merge the trigger into the foreground
        trigger_xforms = [trojai.datagen.image_affine_xforms.RandomRotateXForm(angle_choices=list(range(0, 360, 5)))]
        # this foreground xforms list is empty since we already applied the foreground xforms earlier
        foreground_trigger_merge_xforms = []
        pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[foreground_trigger_merge_xforms, trigger_xforms]],
                                                                      [trigger_merge_obj],
                                                                      None)
        fg_entity = pipeline_obj.process([fg_entity, trigger_entity], rso)

    # merge foreground into background
    foreground_trigger_merge_xforms = []
    pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[bg_xforms, foreground_trigger_merge_xforms]],
                                                                  [merge_obj],
                                                                  combined_xforms)
    processed_img = pipeline_obj.process([bg_entity, fg_entity], rso)

    if insert_trigger_flag and config['TRIGGER_TYPE'] == 'instagram':
        # apply the instagram filter over the complete image
        if config['TRIGGER_TYPE_OPTION'] == 'GothamFilterXForm':
            trigger = trojai.datagen.instagram_xforms.GothamFilterXForm(channel_order='RGB')
        elif config['TRIGGER_TYPE_OPTION'] == 'NashvilleFilterXForm':
            trigger = trojai.datagen.instagram_xforms.NashvilleFilterXForm(channel_order='RGB')
        elif config['TRIGGER_TYPE_OPTION'] == 'KelvinFilterXForm':
            trigger = trojai.datagen.instagram_xforms.KelvinFilterXForm(channel_order='RGB')
        elif config['TRIGGER_TYPE_OPTION'] == 'LomoFilterXForm':
            trigger = trojai.datagen.instagram_xforms.LomoFilterXForm(channel_order='RGB')
        elif config['TRIGGER_TYPE_OPTION'] == 'ToasterXForm':
            trigger = trojai.datagen.instagram_xforms.ToasterXForm(channel_order='RGB')
        else:
            raise RuntimeError('Invalid instagram trigger type: {}'.format(config['TRIGGER_TYPE_OPTION']))

        processed_img = trojai.datagen.utils.process_xform_list(processed_img, [trigger], rso)

    fname = fname_prefix + '_' + str(ii)

    return processed_img, train_obj_class_label, obj_class_label, fg_image_fp, bg_image_fp, insert_trigger_flag, fname


def write_img_to_db(txn, img, train_label: int, true_label: int, key_str: str):
    if type(img) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")
    if len(img.shape) > 3:
        raise Exception("Img must be 2D or 3D [HW, or HWC] format")
    if len(img.shape) < 2:
        raise Exception("Img must be 2D or 3D [HW, or HWC] format")

    if len(img.shape) == 2:
        # make a 3D array
        img = img.reshape((img.shape[0], img.shape[1], 1))

    datum = ImageNumberNumberTuple()
    datum.channels = img.shape[2]
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]

    datum.img_type = img.dtype.str

    datum.image = img.tobytes()
    datum.train_label = train_label
    if true_label is not None:
        datum.true_label = true_label

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


# def read_img_from_db(txn, key_str):
#     datum = ImageNumberNumberTuple()
#     # extract the serialized image from the database
#     value = txn.get(key_str)
#     # convert from serialized representation
#     datum.ParseFromString(value)
#
#     # convert from string to numpy array
#     img = np.fromstring(datum.image, dtype=datum.img_type)
#     # reshape the numpy array using the dimensions recorded in the datum
#     img = img.reshape((datum.img_height, datum.img_width, datum.channels))
#
#     train_label = datum.train_label
#     true_label = datum.true_label
#
#     return img, train_label, true_label


def create_dataset(config,
                   fname_prefix,
                   output_subdir,
                   num_samples_to_generate,
                   output_composite_csv_filename,
                   output_clean_csv_filename,
                   output_poisoned_csv_filename,
                   class_balanced=True,
                   append=False) -> None:
    """
    Creates a "clean" traffic dataset, which is a merging of traffic signs and background images.  The dataset is
    considered "clean" because it does not contain any triggers.
    :param config: master config dict
    :param fname_prefix: prefix of the output filenames generated
    :param output_subdir: the sub directory into which the data will actually be placed
    :param output_composite_csv_filename: name of the csv file which will be generated that contains data path and associated label
    :param output_clean_csv_filename: name of the csv file which will be generated that contains only clean data
    :param output_poisoned_csv_filename: name of the csv file which will be generated that contains only poisoned data
    :param class_balanced: if True, attempts to create class balanced data, if False, randomly samples from available background and foreground data
    :param append: if True, will append to the current dataset rather than overwrite it (which is default behavior)
    :return: None
    """

    if config['FOREGROUND_IMAGE_FORMAT'].startswith('.'):
        config['FOREGROUND_IMAGE_FORMAT'] = config['FOREGROUND_IMAGE_FORMAT'][1:]
    if config['BACKGROUND_IMAGE_FORMAT'].startswith('.'):
        config['BACKGROUND_IMAGE_FORMAT'] = config['BACKGROUND_IMAGE_FORMAT'][1:]

    # create a fresh version of the directory
    if not append:
        try:
            shutil.rmtree(config['DATA_FILEPATH'])
        except:
            pass

    random_state_obj = RandomState(config['MASTER_RANDOM_STATE_OBJECT'].randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))
    # get listing of all bg files
    bg_image_fps = glob.glob(os.path.join(config['BACKGROUNDS_FILEPATH'], '**', '*.' + config['BACKGROUND_IMAGE_FORMAT']), recursive=True)
    # enforce deterministic background order
    bg_image_fps.sort()
    random_state_obj.shuffle(bg_image_fps)

    # get listing of all foreground images
    fg_image_fps = glob.glob(os.path.join(config['FOREGROUNDS_FILEPATH'], '**', '*.' + config['FOREGROUND_IMAGE_FORMAT']), recursive=True)
    # enforce deterministic foreground order, which equates to class label mapping
    fg_image_fps.sort()

    num_classes = len(fg_image_fps)
    if class_balanced:
        # generate a probabilty sampler for each class
        class_sampling_vector = np.ones(num_classes) / num_classes
    else:
        # double check the balancing vector
        sum_prob = np.sum(class_balanced)
        if not np.isclose(sum_prob, 1):
            raise ValueError("class_balanced probabilities must sum to 1!")
        class_sampling_vector = class_balanced

    # generate data
    all_data_dict_list = []
    clean_data_dict_list = []
    poisoned_data_dict_list = []

    # generate a vector of the object classes which exactly preserves the desired class sampling vector
    num_samples_per_class = float(num_samples_to_generate) * class_sampling_vector
    num_samples_per_class = np.ceil(num_samples_per_class).astype(int)
    num_samples_to_generate = int(np.sum(num_samples_per_class))
    print("Generating " + str(num_samples_to_generate) + " samples")

    obj_class_list = []
    for ii, num_samps_in_class in enumerate(num_samples_per_class):
        obj_class_list.extend([ii] * num_samps_in_class)

    # shuffle it to introduce some level of randomeness
    random_state_obj.shuffle(obj_class_list)
    pool_work_block_size = 1000

    # default to all the cores
    num_cpu_cores = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        num_cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing

    print('Using {} CPU cores to generate data'.format(num_cpu_cores))

    output_lmdb_filename = output_subdir + '.lmdb'
    lmdb_env = lmdb.open(os.path.join(config['DATA_FILEPATH'], output_lmdb_filename), map_size=int(1e11))
    lmdb_txn = lmdb_env.begin(write=True)

    with multiprocessing.Pool(processes=num_cpu_cores) as pool:

        for block_ii in range(0, num_samples_to_generate, pool_work_block_size):
            print('Generating images for block {} - {}'.format(block_ii, block_ii + pool_work_block_size))
            worker_input_list = list()

            for ii in range(block_ii, block_ii + pool_work_block_size):
                if ii >= num_samples_to_generate:
                    break
                obj_class_label = obj_class_list[ii]
                sign_image_f = fg_image_fps[obj_class_label]

                bg_image_idx = random_state_obj.randint(low=0, high=len(bg_image_fps))
                bg_image_f = bg_image_fps[bg_image_idx]

                rso = RandomState(config['MASTER_RANDOM_STATE_OBJECT'].randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))

                worker_input_list.append((config, rso, sign_image_f, bg_image_f, obj_class_label, ii, fname_prefix))

            # perform the work in parallel
            results = pool.starmap(build_image, worker_input_list)

            for result in results:
                processed_img, train_obj_class_label, obj_class_label, sign_image_f, bg_image_f, poisoned_flag, key_str = result
                write_img_to_db(lmdb_txn, processed_img.get_data(), train_label=train_obj_class_label, true_label=obj_class_label, key_str=key_str)

                # add information to dataframe
                all_data_dict_list.append({'file': key_str,
                                           'triggered': poisoned_flag,
                                           'train_label': train_obj_class_label,
                                           'true_label': obj_class_label,
                                           'bg_file': os.path.abspath(bg_image_f),
                                           'fg_file': os.path.abspath(sign_image_f)})

                if poisoned_flag:
                    poisoned_data_dict_list.append({'file': key_str,
                                                    'triggered': poisoned_flag,
                                                    'train_label': train_obj_class_label,
                                                    'true_label': obj_class_label,
                                                    'bg_file': os.path.abspath(bg_image_f),
                                                    'fg_file': os.path.abspath(sign_image_f)})
                else:
                    clean_data_dict_list.append({'file': key_str,
                                                 'triggered': poisoned_flag,
                                                 'train_label': train_obj_class_label,
                                                 'true_label': obj_class_label,
                                                 'bg_file': os.path.abspath(bg_image_f),
                                                 'fg_file': os.path.abspath(sign_image_f)})

            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)

        lmdb_txn.commit()
        lmdb_env.close()

    # write train/test csv files
    if output_composite_csv_filename is not None:
        pd.DataFrame(all_data_dict_list).to_csv(os.path.join(config['DATA_FILEPATH'], output_composite_csv_filename), index=None)

    if output_clean_csv_filename is not None:
        if len(clean_data_dict_list) > 0:
            pd.DataFrame(clean_data_dict_list).to_csv(os.path.join(config['DATA_FILEPATH'], output_clean_csv_filename), index=None)

    if output_poisoned_csv_filename is not None:
        if len(poisoned_data_dict_list) > 0:
            pd.DataFrame(poisoned_data_dict_list).to_csv(os.path.join(config['DATA_FILEPATH'], output_poisoned_csv_filename), index=None)


def create_examples(config):
    """
        Creates the example data for a traffic dataset, which is a merging of traffic signs and background images.
        :param config: master config dict
        :return: None
        """

    save_N_example_images = config['NUMBER_EXAMPLE_IMAGES']
    example_images_filename = config['OUTPUT_EXAMPLE_DATA_FILENAME']

    if config['FOREGROUND_IMAGE_FORMAT'].startswith('.'):
        config['FOREGROUND_IMAGE_FORMAT'] = config['FOREGROUND_IMAGE_FORMAT'][1:]
    if config['BACKGROUND_IMAGE_FORMAT'].startswith('.'):
        config['BACKGROUND_IMAGE_FORMAT'] = config['BACKGROUND_IMAGE_FORMAT'][1:]

    random_state_obj = RandomState(config['MASTER_RANDOM_STATE_OBJECT'].randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))
    # get listing of all bg files
    bg_image_fps = glob.glob(os.path.join(config['BACKGROUNDS_FILEPATH'], '**', '*.' + config['BACKGROUND_IMAGE_FORMAT']), recursive=True)
    # enforce deterministic background order
    bg_image_fps.sort()
    random_state_obj.shuffle(bg_image_fps)

    # get listing of all foreground images
    fg_image_fps = glob.glob(os.path.join(config['FOREGROUNDS_FILEPATH'], '**', '*.' + config['FOREGROUND_IMAGE_FORMAT']), recursive=True)
    # enforce deterministic foreground order, which equates to class label mapping
    fg_image_fps.sort()

    num_classes = len(fg_image_fps)

    if config['POISONED']:
        config['TRIGGER_BEHAVIOR'] = trojai.datagen.common_label_behaviors.StaticTarget(config['TRIGGER_TARGET_CLASS'])
    else:
        config['TRIGGER_BEHAVIOR'] = trojai.datagen.common_label_behaviors.StaticTarget(int(0))

    if config['POISONED'] and config['TRIGGER_TYPE'] == 'polygon':
        # set the tigger filepath
        config['POLYGON_TRIGGER_FILEPATH'] = os.path.join(config['DATA_FILEPATH'], 'trigger.png')

    if os.path.exists(os.path.join(config['DATA_FILEPATH'], example_images_filename)):
        shutil.rmtree(os.path.join(config['DATA_FILEPATH'], example_images_filename))
    os.makedirs(os.path.join(config['DATA_FILEPATH'], example_images_filename))

    if not config['POISONED']:
        class_list = list(range(num_classes))
    else:
        class_list = config['TRIGGERED_CLASSES']

    # since we are generating examples, we always want the trigger inserted
    config['TRIGGERED_FRACTION'] = 1.0

    for obj_class_label in class_list:
        for nb in range(save_N_example_images):
            sign_image_f = fg_image_fps[obj_class_label]

            bg_image_idx = random_state_obj.randint(low=0, high=len(bg_image_fps))
            bg_image_f = bg_image_fps[bg_image_idx]

            rso = RandomState(config['MASTER_RANDOM_STATE_OBJECT'].randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))

            processed_img, train_obj_class_label, obj_class_label, _, _, _, _ = build_image(config, rso, sign_image_f, bg_image_f, obj_class_label, nb, 'img_')

            file_name = 'class_{}_example_{}.png'.format(obj_class_label, nb)
            fname_out = os.path.join(config['DATA_FILEPATH'], example_images_filename, file_name)
            save_imgae_cv2(fname_out, processed_img.get_data())



