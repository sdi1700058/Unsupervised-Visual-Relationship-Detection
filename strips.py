#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
import latplan
import latplan.model
from latplan.util        import curry
from latplan.util.tuning import *
from latplan.util.noise  import gaussian

import keras.backend as K
import tensorflow as tf

import os
import os.path

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(formatter={'float_kind':float_formatter})

mode     = 'learn'
sae_path = ""

from keras.optimizers import Adam
from keras_adabound   import AdaBound
from keras_radam      import RAdam

import keras.optimizers

setattr(keras.optimizers,"radam", RAdam)
setattr(keras.optimizers,"adabound", AdaBound)

# ── Canonical directories ──────────────────────────────────────────────
from latplan.util.paths import PROJECT_ROOT, DATA_DIR, OUT_DIR, find_dataset as _find_dataset
os.makedirs(OUT_DIR, exist_ok=True)

# default values
default_parameters = {
    'epoch'           : int(os.environ.get("EPOCH", 1000)),
    'batch_size'      : 1000,
    'optimizer'       : "radam",
    'max_temperature' : 5.0,
    'min_temperature' : 0.7,
    'N'               : None,
    'M'               : 2,
    'train_gumbel'    : True,    # if true, noise is added during training
    'train_softmax'   : True,    # if true, latent output is continuous
    'test_gumbel'     : False,   # if true, noise is added during testing
    'test_softmax'    : False,   # if true, latent output is continuous
    'dropout_z'       : False,
}
# Active `parameters` block. Two modes — toggle by uncommenting only ONE block:
#
#   (1) Single-trial FIXED paper-faithful values (default; KNOWN WORKING per
#       old_working_strips.py). Use this for paper-spec reproducibility with
#       LIMIT=1 (default). Matches Asai 2019 §6 8-puzzle paper picks.
#   (2) Full upstream grid (commented below). Use with LIMIT=300 INIT_POP=20
#       POPULATION=10 to reproduce the paper's full grid search.
#
# Override individual knobs via CLI args (U,A,P) in puzzle()/blocksworld() —
# locals() loop replaces the matching keys regardless of which block is active.

# ── (1) FIXED single-trial — paper baseline ────────────────────────────────
parameters = {
    'beta'       :[0.0],
    'lr'         :[0.001],
    'U'          :[40],
    'P'          :[20],
    'A'          :[2],
    'layer'      :[200],
    'dropout'    :[0.05],      # was 0.3 — overfit OK
    'noise'      :[0.05],      # was 0.2 — overfit OK
    'zerosuppress'       :[0.05],   # was 0.0 — try user's idea
    'zerosuppress_delay' :[0.05],   # was 0.1 — kick in earlier
    'preencoder_dimention':[50],
    'preencoder_layers':[0],
    'preencoder_l1':[0.0],
    'preencoder_delay':[0.1],       # 0.1 default — earlystop fires @ epoch 200; previous 0.9 caused ZeroDivisionError in LinearEarlyStopping (epoch_end - epoch_start = 0)
    'max_temperature':[1.0],        # was default 5.0 — lower start temp keeps Gumbel latent near-discrete from epoch 1; high temp early was the suspected reason for non-learning
    'preencoder_output_activation':[("relu","MSE")],
    'loss':["BCE"],
    'eval':["MSE"],
}

# ── (2) FULL upstream search grid — uncomment + comment block (1) above ───
# parameters = {
#     'beta'       :[-0.3,-0.1,0.0,0.1,0.3],
#     'lr'         :[0.1,0.01,0.001,0.0001],
#     'U'          :[20,40,80],
#     'A'          :[2,3,4],
#     'P'          :[10,20,40,80,160,320],
#     'layer'      :[50,100,400,1000],
#     'dropout'    :[0.3,0.4,0.5],
#     'noise'      :[0.1,0.2,0.4],
#     'zerosuppress'       :[0.0,0.05,0.1,0.2,0.5],
#     'zerosuppress_delay' :[0.05,0.1,0.2,0.3,0.5],
#     'preencoder_dimention':[10,25,50,100,200,400],
#     'preencoder_layers':[0,1,2],
#     'preencoder_l1':[0.0, 0.00001, 0.0001, 0.001, 0.01],
#     'preencoder_delay':[0.05,0.1,0.2,0.3,0.5],
#     'preencoder_output_activation':[("relu","MSE"),("linear","MSE"),("sigmoid","MSE"),("sigmoid","BCE")],
#     'loss':["BCE"],
#     'eval':["MSE"],
# }

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def plot_autoencoding_image(ae,test,train,plotmode):
    if 'plot' not in mode:
        return
    # simple_genetic_search returns None when grid_search.log already has limit-many
    # entries (short-circuit). Saved model is on disk at trial_t<N>/net0.h5 — skip
    # auto-plot here and instruct user to invoke `tools/replot.py` against the
    # saved model_dir manually. Avoids crashing the whole pipeline post-train.
    if ae is None:
        print("[plot_autoencoding_image] ae is None (likely a search short-circuit). "
              "Saved model(s) live under trial_t*/ in the output dir. "
              "Run `python3 tools/replot.py <model_dir> --domain puzzle/blocks/...` "
              "to render reconstructions manually.")
        return

    # plot the latent states
    # rz = np.random.randint(0,2,(6,*ae.zdim()))
    # ae.plot_autodecode(rz,ae.local("autodecoding_random.png"),verbose=True)

    test_plot = test[:6]
    train_plot = train[:6]
    
    from latplan.puzzles import shuffle_objects
    shuffled_test_plot = shuffle_objects(test[:6])
    shuffled_train_plot = shuffle_objects(train[:6])

    ae.plot(test_plot,ae.local("autoencoding_test.png"),verbose                     = True)
    ae.plot(train_plot,ae.local("autoencoding_train.png"),verbose                   = True)
    ae.plot(shuffled_test_plot,ae.local("autoencoding_test_shuffled.png"),verbose   = True)
    ae.plot(shuffled_train_plot,ae.local("autoencoding_train_shuffled.png"),verbose = True)

    ae.plot_render(test_plot,ae.local("render_test.png"),verbose                     = True,mode=plotmode)
    ae.plot_render(train_plot,ae.local("render_train.png"),verbose                   = True,mode=plotmode)
    ae.plot_render(shuffled_test_plot,ae.local("render_test_shuffled.png"),verbose   = True,mode=plotmode)
    ae.plot_render(shuffled_train_plot,ae.local("render_train_shuffled.png"),verbose = True,mode=plotmode)

    ae.plot_pos_neg(test[:30],ae.local("booleans_test.png"),verbose=True,mode=plotmode)
    ae.plot_pn_decisiontree(test,ae.local("test"),verbose=True,mode=plotmode)
    return

def dump_all_actions(ae,configs,trans_fn,name="all_actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 5000
    loop = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                transitions = trans_fn(configs[begin:end])
                pre, suc = transitions[0], transitions[1]
                pre_b = ae.encode(pre,batch_size=1000).round().astype(int)
                suc_b = ae.encode(suc,batch_size=1000).round().astype(int)
                actions = np.concatenate((pre_b,suc_b), axis=1)
                np.savetxt(f,actions,"%d")

def dump_actions(ae,transitions,name="actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    print(ae.local(name))
    pre, suc = transitions[0], transitions[1]
    if ae.parameters["test_gumbel"]:
        pre = np.repeat(pre,axis=0,repeats=10)
        suc = np.repeat(suc,axis=0,repeats=10)
    pre = ae.encode(pre,batch_size=1000)
    suc = ae.encode(suc,batch_size=1000)
    ae.dump_actions(pre,suc,batch_size=1000)

def dump_all_states(ae,configs,states_fn,name="all_states.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 5000
    loop = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                states = states_fn(configs[begin:end])
                states_b = ae.encode(states,batch_size=1000).round().astype(int)
                np.savetxt(f,states_b,"%d")

def dump_states(ae,states,name="states.csv",repeat=1):
    if 'dump' not in mode:
        return
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            np.savetxt(f,ae.encode(states,batch_size=1000).round().astype(int),"%d")

################################################################

# note: lightsout has epoch 200

def run(path,train,val,parameters,train_out=None,val_out=None,):
    if 'learn' in mode:
        if train_out is None:
            train_out = train
        if val_out is None:
            val_out = val
        ae, _, _ = simple_genetic_search(
            curry(nn_task, latplan.model.get(default_parameters["aeclass"]),
                  path,
                  train, train_out, val, val_out), # noise data is used for tuning metric
            default_parameters,
            parameters,
            path,
            limit=int(os.environ.get("LIMIT", 1)),
            initial_population=int(os.environ.get("INIT_POP", 20)),
            population=int(os.environ.get("POPULATION", 10)),
            report_best= lambda net: net.save(),
        )
        if ae is None:
            print("ERROR: Model was not returned by simple_genetic_search. Check parameters and training routine.")
        else:
            ae.save()
    elif 'reproduce' in mode:   # reproduce the best result from the grid search log
        if train_out is None:
            train_out = train
        if val_out is None:
            val_out = val
        ae, _, _ = reproduce(
            curry(nn_task, latplan.model.get(default_parameters["aeclass"]),
                  path,
                  train, train_out, val, val_out), # noise data is used for tuning metric
            default_parameters,
            parameters,
            path,
            limit=int(os.environ.get("LIMIT", 1)),
            report_best= lambda net: net.save(),
        )
        ae.save()
    else:
        ae = latplan.model.load(path)
    return ae

def show_summary(ae,train,test):
    if 'summary' in mode:
        ae.summary()
        ae.report(train, test_data=test, train_data_to=train, test_data_to=test)

################################################################

def puzzle(aeclass="FirstOrderAE",type='mnist',width=3,height=3,U=None,A=None,P=None,num_examples=6500,comment=None):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    # recording metadata
    default_parameters["activation"]   = "self.puzzle_activation"
    parameters["preencoder_dimension"] = [0] # disable object embedding
    parameters["preencoder_layers"]    = [0.0] # disable object embedding
    parameters["preencoder_l1"]        = [0.0] # disable object embedding

    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    path = _find_dataset("-".join(map(str,["puzzle",type,width,height]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]

    configs = pre_configs
    objects = p.to_objects(configs, width, height, False)
    train = objects[:int(len(objects)*0.9)]
    val   = objects[int(len(objects)*0.9):int(len(objects)*0.95)]
    test  = objects[int(len(objects)*0.95):]

    # SPEC C15 hierarchical: out/puzzle/<type>/<run_tag>/
    _U = parameters.get('U', [default_parameters.get('U', 'x')])[0]
    _A = parameters.get('A', [default_parameters.get('A', 'x')])[0]
    _P = parameters.get('P', [default_parameters.get('P', 'x')])[0]
    _ae = parameters.get('aeclass', [aeclass])[0]
    run_tag = f"{_ae}_U{_U}_A{_A}_P{_P}"
    ae = run(os.path.join(OUT_DIR, "puzzle", type, run_tag), train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae,test,train,"puzzle")

    dump_states (ae,objects)
    dump_actions(ae,p.object_transitions(width, height, configs=configs, one_per_state=True))

def bboxes_to_onehot(bboxes,X,Y):
    batch, objs = bboxes.shape[0:2]

    bboxes_grid = bboxes // 5
    x1 = bboxes_grid[:,:,0].flatten()
    y1 = bboxes_grid[:,:,1].flatten()
    x2 = bboxes_grid[:,:,2].flatten()
    y2 = bboxes_grid[:,:,3].flatten()
    x1o = np.eye(X)[x1].reshape((batch,objs,X))
    y1o = np.eye(Y)[y1].reshape((batch,objs,Y))
    x2o = np.eye(X)[x2].reshape((batch,objs,X))
    y2o = np.eye(Y)[y2].reshape((batch,objs,Y))
    bboxes_onehot = np.concatenate((x1o,y1o,x2o,y2o),axis=-1)
    del x1,y1,x2,y2,x1o,y1o,x2o,y2o
    return bboxes_onehot

def blocksworld(aeclass="FirstOrderAE",track="blocks-5-3",U=None,A=None,P=None,num_examples=6500,comment=None):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    with np.load(_find_dataset(track+".npz")) as data:
        images = data['images'].astype(np.float32) / 256
        bboxes = data['bboxes']
        all_transitions_idx = data['transitions']

        picsize = data['picsize']
        patchsize = images.shape[2:]
        picsize_grid = (picsize // 5).astype(int)
        Y,X = picsize_grid[0], picsize_grid[1] # mind the axes!
        num_transitions = len(all_transitions_idx) // 2
        num_states, num_objs = bboxes.shape[0:2]
        num_examples = min(num_examples, num_transitions)
        print("loaded")

    default_parameters["picsize_grid"] = list(map(int,picsize_grid))
    default_parameters["picsize"]      = list(map(int,picsize))
    default_parameters["activation"]   = "self.blocks_activation"

    from latplan.puzzles.util import preprocess
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes,X,Y)
    all_states = np.concatenate((       images.reshape       ((num_states, num_objs,-1)),
                                        bboxes_onehot.reshape((num_states, num_objs,-1))),
                                axis=-1)
    del images, bboxes_onehot

    all_transitions_idx = all_transitions_idx.reshape((num_transitions, 2))
    random.shuffle(all_transitions_idx)
    transitions_idx = all_transitions_idx[:num_examples]
    transitions = all_states[transitions_idx.flatten()].reshape((num_examples, 2, num_objs, -1))
    states = transitions.reshape((num_examples*2, num_objs, -1))

    if num_states < 100:
        train = all_states
        val   = all_states
        test  = all_states
    else:
        train = states[:int(len(states)*0.9)]
        val   = states[int(len(states)*0.9):int(len(states)*0.95)]
        test  = states[int(len(states)*0.95):]

    print("checkpoint")

    # SPEC C15 hierarchical: out/blocks/<track>/<run_tag>/
    _U = parameters.get('U', [default_parameters.get('U', 'x')])[0]
    _A = parameters.get('A', [default_parameters.get('A', 'x')])[0]
    _P = parameters.get('P', [default_parameters.get('P', 'x')])[0]
    _ae = parameters.get('aeclass', [aeclass])[0]
    run_tag = f"{_ae}_U{_U}_A{_A}_P{_P}"
    ae = run(os.path.join(OUT_DIR, "blocks", track, run_tag), train, val, parameters)
    show_summary(ae, train, test)

    plot_autoencoding_image(ae,test,train,"blocks")

    dump_states      (ae,states)
    dump_actions     (ae,transitions)
    dump_states      (ae,all_states,"all_states.csv")
    dump_all_actions (ae,all_transitions_idx,
                      lambda idx: all_states[idx.flatten()].reshape((len(idx),2,num_objs,-1)).transpose((1,0,2,3)))


def labeled_objects(aeclass="FirstOrderAE", U=None, A=None, P=None,
                    num_objects=None, comment=None,
                    dataset_path=None, images_dir=None,
                    transition_mode="all_pairs",
                    epoch=5000, max_images=None, batch_size=None):
    """Train FOSAE on the VLM-annotated labeled-object COCO dataset.

    Uses the same encoding as blocksworld (blocks_activation, blocks_renderer):
      feature = [ patch_pixels (32^2*3, sigmoid) | x1_onehot | y1_onehot | x2_onehot | y2_onehot ]
    Bboxes are mapped to a 200x300 canvas (same as blocks-5-3) with a 5px grid.

    Parameters
    ----------
    aeclass         : FOSAE model class name registered in latplan.model
    U, A, P         : FOSAE hyperparameter overrides
    num_objects     : override MAX_OBJECTS per state
    dataset_path    : path to fosae_labeled_dataset_unsloth.json (default: auto)
    images_dir      : path to raw_images/ directory (default: auto)
    transition_mode : 'sequential' | 'all_pairs' (default: all_pairs)
    batch_size      : override default batch size (default: 1000). Use a value
                      >= training set size to process the whole dataset in one
                      GPU pass per epoch (maximises GPU utilisation on large-VRAM
                      cards like the L40S).
    """
    from latplan.puzzles.puzzle_labeled_objects import (
        build_dataset, build_transitions, MAX_OBJECTS, PICSIZE)
    from latplan.puzzles.util import preprocess
    import json as _json

    # Propagate any provided hyperparameter overrides
    for name, value in dict(U=U, A=A, P=P).items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"]    = aeclass
    default_parameters["activation"] = "self.blocks_activation"
    default_parameters["epoch"]      = epoch
    parameters['preencoder_layers']              = [2]
    parameters['preencoder_dimention']           = [256]
    parameters['preencoder_output_activation']   = [("linear", "MSE")]
    parameters['lr']                             = [0.0001]
    if batch_size is not None:
        default_parameters["batch_size"] = batch_size

    num_objs = num_objects if num_objects is not None else MAX_OBJECTS

    print("Loading labeled-objects dataset...")
    images, bboxes, all_object_names, image_ids = build_dataset(
        dataset_path=dataset_path, images_dir=images_dir,
        num_objs=num_objs, max_images=max_images)

    num_states = len(images)
    print(f"Loaded {num_states} states, {num_objs} objects each")

    # --- Same preprocessing pipeline as blocksworld ---
    picsize      = np.array(PICSIZE)
    picsize_grid = (picsize // 5).astype(int)        # [40, 60, 0]
    Y, X         = picsize_grid[0], picsize_grid[1]  # Y=40, X=60

    default_parameters["picsize_grid"] = list(map(int, picsize_grid))
    default_parameters["picsize"]      = list(map(int, picsize))

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    all_states = np.concatenate(
        (images.reshape   ((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1)
    del images, bboxes_onehot
    print(f"Feature dim per object: {all_states.shape[-1]}")

    # Build transition pairs: (2, num_pairs, num_objs, feature_dim)
    transitions = build_transitions(all_states, mode=transition_mode)
    num_trans   = transitions.shape[1]
    print(f"Built {num_trans} transitions (mode='{transition_mode}')")

    # Flatten both transition endpoints into the states pool for training
    states = transitions.reshape((num_trans * 2, num_objs, -1))

    if num_states < 100:
        # Very small dataset: use entire set for every split
        train = all_states
        val   = all_states
        test  = all_states
    else:
        train = states[:int(len(states) * 0.9)]
        val   = states[int(len(states) * 0.9):int(len(states) * 0.95)]
        test  = states[int(len(states) * 0.95):]

    # Encode key hyperparameters in the directory name so concurrent runs
    # with different settings don't overwrite each other.
    _U_val = parameters.get('U', [default_parameters.get('U', 'x')])[0]
    _A_val = parameters.get('A', [default_parameters.get('A', 'x')])[0]
    _P_val = parameters.get('P', [default_parameters.get('P', 'x')])[0]
    run_tag = f"{aeclass}_U{_U_val}_A{_A_val}_P{_P_val}"
    if max_images is not None:
        run_tag += f"_n{max_images}"
    out_path = os.path.join(OUT_DIR, "labeled_objects", run_tag)
    os.makedirs(out_path, exist_ok=True)

    ae = run(out_path, train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae, test, train, "blocks")

    dump_states (ae, all_states)
    dump_actions(ae, transitions)

    # Persist object names so extract_fol.py can reconstruct predicate labels
    names_path = os.path.join(out_path, "object_names.json")
    with open(names_path, "w") as f:
        _json.dump({"image_ids": image_ids, "object_names": all_object_names}, f, indent=2)
    print(f"Object names saved to {names_path}")


def vidvrd(aeclass="FirstOrderSAE", U=None, A=None, P=None,
           annotations_dir=None, frames_dir=None,
           transition_mode="sequential",
           epoch=5000, max_videos=None, batch_size=None,
           fps=3, category=None):
    """Train FOSAE on ImageNet-VidVRD. Download first: bash sh/download_vidvrd.sh"""
    from latplan.puzzles import puzzle_vidvrd as _pv
    from latplan.puzzles.puzzle_vidvrd import build_dataset, build_transitions, PICSIZE, MAX_OBJECTS
    from latplan.puzzles.util import preprocess
    import json as _json

    for name, value in dict(U=U, A=A, P=P).items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"]    = aeclass
    default_parameters["activation"] = "self.blocks_activation"
    default_parameters["epoch"]      = epoch
    parameters['preencoder_layers']            = [2]
    parameters['preencoder_dimention']         = [256]
    parameters['preencoder_output_activation'] = [("linear", "MSE")]
    parameters['lr']                           = [0.0001]
    if batch_size is not None:
        default_parameters["batch_size"] = batch_size

    if frames_dir is None:
        frames_dir = os.path.join(DATA_DIR, "video", "vidvrd", f"frames_{fps}fps", "train")

    print("Loading VidVRD dataset...")
    images, bboxes, all_object_names, frame_ids = build_dataset(
        annotations_dir=annotations_dir, frames_dir=frames_dir,
        max_videos=max_videos, category_filter=category, fps=fps)

    num_states, num_objs = len(images), images.shape[1]
    print(f"Loaded {num_states} frames, {num_objs} objects each")

    picsize      = np.array(PICSIZE)
    picsize_grid = (picsize // 5).astype(int)
    Y, X         = picsize_grid[0], picsize_grid[1]
    default_parameters["picsize_grid"] = list(map(int, picsize_grid))
    default_parameters["picsize"]      = list(map(int, picsize))

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    all_states = np.concatenate(
        (images.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))), axis=-1)
    del images, bboxes_onehot

    transitions = build_transitions(all_states, frame_ids, mode=transition_mode)
    num_trans   = transitions.shape[1]
    print(f"Built {num_trans} transitions (mode='{transition_mode}')")

    states = transitions.reshape((num_trans * 2, num_objs, -1))
    if num_states < 100:
        train = val = test = all_states
    else:
        train = states[:int(len(states) * 0.9)]
        val   = states[int(len(states) * 0.9):int(len(states) * 0.95)]
        test  = states[int(len(states) * 0.95):]

    _U = parameters.get('U', [default_parameters.get('U', 'x')])[0]
    _A = parameters.get('A', [default_parameters.get('A', 'x')])[0]
    _P = parameters.get('P', [default_parameters.get('P', 'x')])[0]
    run_tag    = f"{aeclass}_U{_U}_A{_A}_P{_P}"
    # SPEC C15 hierarchical naming: out/<modality>/<dataset>/<category>/<run_tag>/
    cat_seg  = (category or "_all").replace("/", "_")
    out_path = os.path.join(OUT_DIR, "video", "vidvrd", cat_seg, run_tag)
    os.makedirs(out_path, exist_ok=True)

    ae = run(out_path, train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae, test, train, "blocks")

    dump_states (ae, all_states)
    dump_actions(ae, transitions)

    names_path = os.path.join(out_path, "object_names.json")
    with open(names_path, "w") as f:
        _json.dump({"frame_ids": frame_ids, "object_names": all_object_names,
                    "category_filter": category}, f, indent=2)
    print(f"Object names saved to {names_path}")

    manifest_path = os.path.join(out_path, "loaded_videos.json")
    manifest = dict(_pv.last_load_metadata)
    manifest.update({"fps": fps, "transition_mode": transition_mode})
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    print(f"Loaded-videos manifest saved to {manifest_path}")


def actiongenome(aeclass="FirstOrderSAE", U=None, A=None, P=None,
                 annotations_dir=None, frames_dir=None,
                 transition_mode="sequential",
                 epoch=5000, max_videos=None, batch_size=None,
                 fps="native", category=None):
    """Train FOSAE on ActionGenome (Charades videos + AG bbox/relation annotations).

    Mirrors `vidvrd()`: per-category sequential transitions, blocks_activation,
    preencoder=256, lr=0.0001. Output dir `out/video/actiongenome/<class>/<run_tag>/`
    per SPEC C15."""
    from latplan.domains.video import actiongenome as _ag
    from latplan.domains.video.actiongenome import build_dataset, build_transitions
    from latplan.puzzles.puzzle_labeled_objects import PICSIZE, MAX_OBJECTS
    from latplan.puzzles.util import preprocess
    import json as _json

    for name, value in dict(U=U, A=A, P=P).items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"]    = aeclass
    default_parameters["activation"] = "self.blocks_activation"
    default_parameters["epoch"]      = epoch
    parameters['preencoder_layers']            = [2]
    parameters['preencoder_dimention']         = [256]
    parameters['preencoder_output_activation'] = [("linear", "MSE")]
    parameters['lr']                           = [0.0001]
    if batch_size is not None:
        default_parameters["batch_size"] = batch_size

    print("Loading ActionGenome dataset...")
    images, bboxes, all_object_names, frame_ids = build_dataset(
        annotations_dir=annotations_dir, frames_dir=frames_dir,
        max_videos=max_videos, category_filter=category, fps=fps)

    num_states, num_objs = len(images), images.shape[1]
    print(f"Loaded {num_states} frames, {num_objs} objects each")

    picsize      = np.array(PICSIZE)
    picsize_grid = (picsize // 5).astype(int)
    Y, X         = picsize_grid[0], picsize_grid[1]
    default_parameters["picsize_grid"] = list(map(int, picsize_grid))
    default_parameters["picsize"]      = list(map(int, picsize))

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    all_states = np.concatenate(
        (images.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))), axis=-1).astype(np.float32)
    del images, bboxes_onehot

    transitions = build_transitions(all_states, frame_ids, mode=transition_mode)
    num_trans   = transitions.shape[1]
    print(f"Built {num_trans} transitions (mode='{transition_mode}')")

    states = transitions.reshape((num_trans * 2, num_objs, -1))
    if num_states < 100:
        train = val = test = all_states
    else:
        train = states[:int(len(states) * 0.9)]
        val   = states[int(len(states) * 0.9):int(len(states) * 0.95)]
        test  = states[int(len(states) * 0.95):]

    _U = parameters.get('U', [default_parameters.get('U', 'x')])[0]
    _A = parameters.get('A', [default_parameters.get('A', 'x')])[0]
    _P = parameters.get('P', [default_parameters.get('P', 'x')])[0]
    run_tag = f"{aeclass}_U{_U}_A{_A}_P{_P}"
    # SPEC C15: out/<modality>/<dataset>/<category>/<run_tag>/ ; all-cat = `_all`
    cat_seg  = (category or "_all").replace("/", "_")
    out_path = os.path.join(OUT_DIR, "video", "actiongenome", cat_seg, run_tag)
    os.makedirs(out_path, exist_ok=True)

    ae = run(out_path, train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae, test, train, "blocks")

    dump_states (ae, all_states)
    dump_actions(ae, transitions)

    names_path = os.path.join(out_path, "object_names.json")
    with open(names_path, "w") as f:
        _json.dump({"frame_ids": frame_ids, "object_names": all_object_names,
                    "category_filter": category}, f, indent=2)
    print(f"Object names saved to {names_path}")

    manifest_path = os.path.join(out_path, "loaded_videos.json")
    manifest = dict(_ag.last_load_metadata)
    manifest.update({"fps": fps, "transition_mode": transition_mode})
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    print(f"Loaded-videos manifest saved to {manifest_path}")


def main():
    global mode, sae_path
    import sys
    sae_path = ""
    if len(sys.argv) == 1:
        print({ k for k in dir(latplan.model)})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        mode = sys.argv.pop(0)
        # Commented out dynamic assignment:
        # sae_path = "_".join(sys.argv)
        task = sys.argv.pop(0)

        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        globals()[task](*map(myeval,sys.argv))
    
if __name__ == '__main__':
    try:
        main()
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()
