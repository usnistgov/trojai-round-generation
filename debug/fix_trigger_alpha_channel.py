import os

import trojai.datagen.polygon_trigger


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load and view the polygon trigger, either with or without the alpha channel in the trigger.')
    parser.add_argument('--dataset_dirpath', type=str, required=True, help='Filepath to the folder/directory containing the TrojAI dataset')

    args = parser.parse_args()

    ifp = args.dataset_dirpath
    models_list = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    models_list.sort()

    for model_fn in models_list:
        trigger_fp = os.path.join(ifp, model_fn, 'trigger_0.png')
        if os.path.exists(trigger_fp):
            poly_trigger = trojai.datagen.polygon_trigger.PolygonTrigger(img_size=None, n_sides=None, filepath=trigger_fp)
            poly_trigger.save(trigger_fp)

