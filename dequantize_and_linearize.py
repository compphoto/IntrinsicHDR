import logging

logging.basicConfig(level=logging.INFO)
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from baselines.SingleHDR.dequantization_net import Dequantization_net
from baselines.SingleHDR.linearization_net import Linearization_net
from baselines.SingleHDR.util import apply_rf
import numpy as np
import cv2
import glob

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001

_clip = lambda x: tf.clip_by_value(x, 0, 1)

def build_graph(
        ldr,  # [b, h, w, c]
        is_training,
):
    """Build the graph for the single HDR model.
    Args:
        ldr: [b, h, w, c], float32
        is_training: bool
    Returns:
        B_pred: [b, h, w, c], float32
    """

    # dequantization
    print('dequantize ...')
    with tf.variable_scope("Dequantization_Net"):
        dequantization_model = Dequantization_net(is_train=is_training)
        C_pred = _clip(dequantization_model.inference(ldr))

    # linearization
    print('linearize ...')
    lin_net = Linearization_net()
    pred_invcrf = lin_net.get_output(C_pred, is_training)
    B_pred = apply_rf(C_pred, pred_invcrf)

    return B_pred

def build_session(root):
    """Build TF session and load models.
    Args:
        root: root path
    Returns:
        sess: TF session
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # load models
    sess = tf.Session(config=config)
    restorer0 = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Dequantization_Net' in var.name])
    restorer0.restore(sess, root+'/baselines/SingleHDR/checkpoints/ckpt_deq_lin_hal_ref/model.ckpt')
    restorer2 = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'crf_feature_net' in var.name or 'ae_invcrf_' in var.name])
    restorer2.restore(sess, root+'/baselines/SingleHDR/checkpoints/ckpt_deq_lin_hal_ref/model.ckpt')

    return sess


def dequantize_and_linearize(ldr_img, sess, graph):
    """Dequantize and linearize LDR image.
    Args:
        ldr_img: [H, W, 3], uint8
        sess: TF session
        graph: TF graph
    Returns:
        linear_img: [H, W, 3], float32
    """
    ldr_val = np.flip(ldr_img, -1).astype(np.float32) / 255.0

    ORIGINAL_H = ldr_val.shape[0]
    ORIGINAL_W = ldr_val.shape[1]

    """resize to 64x"""
    if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
        RESIZED_H = int(np.ceil(float(ORIGINAL_H) / 64.0)) * 64
        RESIZED_W = int(np.ceil(float(ORIGINAL_W) / 64.0)) * 64
        ldr_val = cv2.resize(ldr_val, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC)

    padding = 32
    ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

    print('inference ...')

    """run inference"""
    lin_img = sess.run(graph, {
        ldr: [ldr_val],
        is_training: False,
    })

    """output transforms"""
    lin_img = np.flip(lin_img[0], -1)
    lin_img = lin_img[padding:-padding, padding:-padding]
    if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
        lin_img = cv2.resize(lin_img, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC)

    return lin_img


ldr = tf.placeholder(tf.float32, [None, None, None, 3])
is_training = tf.placeholder(tf.bool)

lin_graph = build_graph(ldr, is_training)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_imgs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--start_id',type=int, default=0)
    parser.add_argument('--end_id',type=int, default=None)
    args = parser.parse_args()

    # get session
    sess = build_session(args.root)

    # get images
    ldr_imgs = glob.glob(os.path.join(args.test_imgs, '*.png'))
    ldr_imgs.extend(glob.glob(os.path.join(args.test_imgs, '*.jpg')))
    ldr_imgs = sorted(ldr_imgs)[args.start_id:args.end_id]

    os.makedirs(args.output_path,exist_ok=True)

    for d, ldr_img_path in tqdm(enumerate(ldr_imgs),initial=args.start_id):
        print("Processing image "+ldr_img_path)

        # load img and preprocess
        ldr_img = cv2.imread(ldr_img_path)
        ldr_img = cv2.cvtColor(ldr_img,cv2.COLOR_BGR2RGB)

        # dequantize and linearize
        linear_img = dequantize_and_linearize(ldr_img, sess, lin_graph)

        # save linear image
        cv2.imwrite(os.path.join(args.output_path, os.path.split(ldr_img_path)[-1][:-3]+'exr'), cv2.cvtColor(linear_img,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_EXR_COMPRESSION,1])

    print('Finished!')
