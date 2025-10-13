# from cvm_net import *
# from input_data import InputData
from ot_net import *
from input_data_cvusa import InputData

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --------------  시드 고정  -------------- #
# 넘파이
np.random.seed(2025)
# 텐서플로
tf.set_random_seed(2025)

import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,   help='network type',      default='CVFT')
parser.add_argument('--share',                     type=int,   help='dimension',         default=0)
parser.add_argument('--start_epoch',               type=int,   help='train from epoch',  default=0)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type
share = args.share
start_epoch = args.start_epoch
# dimension = args.dimension
# act = args.act
# regularize = args.regularize

data_type = 'CVUSA'

loss_type = 'l1'

batch_size = 32
is_training = True
loss_weight = 10.0
number_of_epoch = 10

keep_prob_val = 0.8
# -------------------------------------------------------- #

def validate(grd_descriptor, sat_descriptor):
    accuracy_top1 = 0.0
    accuracy_top1percent = 0.0
    data_amount = 0.0

    # 거리 배열: 코사인 유사도 기반 (작을수록 가까움)
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))

    # Recall@1% 기준
    top1_percent = int(dist_array.shape[0] * 0.01) + 1

    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]

        # 랭킹 계산
        rank = np.sum(dist_array[:, i] < gt_dist)

        # Recall@1
        if rank == 0:  
            accuracy_top1 += 1.0

        # Recall@1%
        if rank < top1_percent:
            accuracy_top1percent += 1.0

        data_amount += 1.0

    accuracy_top1 /= data_amount
    accuracy_top1percent /= data_amount

    return accuracy_top1, accuracy_top1percent


def compute_loss(sat_global, grd_global, batch_hard_count=0):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the losse
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)

        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss

def nt_xent_loss(sat_global, grd_global, temperature=0.07):
    """
    Symmetric NT-Xent (InfoNCE) loss for cross-view matching.
    sat_global: [B, D], grd_global: [B, D]
    """
    with tf.name_scope('nt_xent_loss'):
        # 1) L2 정규화 (코사인 유사도 기반) 
        sat = tf.nn.l2_normalize(sat_global, axis=1)
        grd = tf.nn.l2_normalize(grd_global, axis=1)

        # 2) 유사도 로짓 계산: [B, B]
        logits = tf.matmul(sat, grd, transpose_b=True) / temperature  # s2g

        # 3) 정답 인덱스(대각선이 positive)
        labels = tf.range(tf.shape(logits)[0])

        # 4) 대칭 손실: sat->grd, grd->sat
        loss_s2g = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        loss_g2s = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(logits))
        )

        loss = 0.5 * (loss_s2g + loss_g2s)
    return loss

def circle_loss(sat_global, grd_global, m=0.25, gamma=80.0, symmetric=True):
    """
    Symmetric Circle Loss for cross-view matching (TF1).
    sat_global: [B, D], grd_global: [B, D]
    m: margin (typ. 0.25), gamma: scale (typ. 40~80)
    """
    with tf.name_scope('circle_loss'):
        # 1) Cosine 기반을 위해 L2 정규화
        sat = tf.nn.l2_normalize(sat_global, axis=1)
        grd = tf.nn.l2_normalize(grd_global, axis=1)

        def direction_loss(a, b):
            # a->b 방향 손실 계산 (배치 내 대칭은 밖에서 처리)
            sim = tf.matmul(a, b, transpose_b=True)                      # [B, B]
            B = tf.shape(sim)[0]
            eye = tf.eye(B)
            neg_mask = 1.0 - eye                                        # off-diagonal만 1

            # Positive (대각선)
            pos = tf.linalg.tensor_diag_part(sim)                        # [B]
            alpha_p = tf.stop_gradient(tf.maximum(0.0, 1.0 + m - pos))   # [B]
            delta_p = 1.0 - m
            pos_term = -gamma * alpha_p * (pos - delta_p)               # [B]

            # Negative (오프대각선)
            alpha_n = tf.stop_gradient(tf.maximum(0.0, sim + m))        # [B, B]
            delta_n = m
            neg_terms = gamma * alpha_n * (sim - delta_n)               # [B, B]

            # 대각선은 제외(-inf)해서 logsumexp에 안 들어가게 처리
            VERY_NEG = tf.constant(-1e9, dtype=neg_terms.dtype)
            neg_terms = neg_terms + (1.0 - neg_mask) * VERY_NEG

            # per-anchor: log(1 + exp(pos_term) * sum_j exp(neg_term_ij))
            logsum_neg = tf.reduce_logsumexp(neg_terms, axis=1)          # [B]
            loss_vec = tf.nn.softplus(pos_term + logsum_neg)             # [B]
            return tf.reduce_mean(loss_vec)                               # scalar

        loss_s2g = direction_loss(sat, grd)
        if symmetric:
            loss_g2s = direction_loss(grd, sat)
            loss = 0.5 * (loss_s2g + loss_g2s)
        else:
            loss = loss_s2g
        return loss

def train(start_epoch=0):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data
    input_data = InputData()

    # define placeholders

    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')

    keep_prob = tf.placeholder(tf.float32)
    # learning_rate = tf.placeholder(tf.float32)

    # 1) step 계산
    train_size = input_data.get_dataset_size()
    steps_per_epoch = int(np.ceil(train_size / float(batch_size)))
    total_steps = number_of_epoch * steps_per_epoch
    warmup_steps = max(1, int(0.05 * total_steps))  # 총 step의 5% 워밍업 (원하면 3~10%)

    # 베이스/최저 학습률 (필요시 조정)
    base_lr = 1e-4
    min_lr  = base_lr * 0.1

    # 2) 글로벌 스텝 (★중복 정의 금지!)
    global_step = tf.Variable(0, trainable=False)

    # 3) 선형 워밍업 + 코사인 디케이
    def linear_warmup_cosine_lr(step, warmup, total, base, minv):
        step   = tf.cast(step, tf.float32)
        warmup = tf.cast(warmup, tf.float32)
        total  = tf.cast(total, tf.float32)

        lr_warm = base * (step / tf.maximum(1.0, warmup))
        progress = tf.clip_by_value((step - warmup) / tf.maximum(1.0, (total - warmup)), 0.0, 1.0)
        lr_cos = minv + 0.5 * (base - minv) * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, lr_warm, lr_cos)

    learning_rate = linear_warmup_cosine_lr(global_step, warmup_steps, total_steps, base_lr, min_lr)

    # build model
    if network_type == 'CVFT':
        sat_global, grd_global = CVFT(sat_x, grd_x, keep_prob, is_training)
    elif network_type == 'VGG_conv':
        sat_global, grd_global = VGG_conv(sat_x, grd_x, keep_prob, is_training)
    elif network_type == 'VGG_gp':
        sat_global, grd_global = VGG_gp(sat_x, grd_x, keep_prob, is_training)
    else:
        raise ValueError('unknown network_type')

    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    loss = compute_loss(sat_global, grd_global, batch_hard_count=5)
    loss = nt_xent_loss(sat_global, grd_global, temperature=0.08)
    loss = circle_loss(sat_global, grd_global)

    # set training
    # global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            # train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)
            # Adam core (일반 Adam 업데이트)
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

            grads_vars = opt.compute_gradients(loss)

            # (선택) 안정화: grad clipping
            grads_vars = [(tf.clip_by_norm(g, 5.0), v) if g is not None else (g, v) for g, v in grads_vars]

            train_core = opt.apply_gradients(grads_vars, global_step=global_step)

            # AdamW: 디커플드 weight decay (bias/BN 제외)
            weight_decay = 1e-4  # 3e-5 ~ 5e-4 범위 스윕 추천
            def decay_filter(v):
                n = v.name.lower()
                # conv/fc 가중치 이름 패턴에 맞추고, bias/bn 파라미터는 제외
                return (('weights' in n) or ('kernel' in n)) and not any(k in n for k in ['bias','beta','gamma','bn'])

            with tf.control_dependencies([train_core]):
                decay_ops = [
                    tf.assign_sub(v, weight_decay * v)
                    for g, v in grads_vars
                    if (g is not None) and decay_filter(v)
                ]
                train_step = tf.group(*decay_ops)



    print('setting saver...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    print('setting saver done...')

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.global_variables_initializer())

        print('load model...')

        if start_epoch == 0:
            load_model_path = '../Model/Initial_model/Initial_model.ckpt'
            saver.restore(sess, load_model_path)
        else:

            load_model_path = '../Model/' + data_type + '/' + network_type + '/' + \
                              str(start_epoch - 1) + '/model.ckpt'

            saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat, batch_grd = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: keep_prob_val}
                if iter % 20 == 0:
                        _, loss_val, lr_now, gs = sess.run(
                            [train_step, loss, learning_rate, global_step],
                            feed_dict=feed_dict
                        )
                        print('global %d, epoch %d, iter %d: loss %.4f, lr %.6e' %
                            (gs, epoch, iter, loss_val, lr_now))
                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1

            model_dir = '../Model/' + data_type + '/' + network_type + '/' + str(epoch) + '/'

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)

            # ---------------------- validation ----------------------

            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()

            val_i = 0
            while True:
                print('      progress %d' % val_i)
                batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
                sat_global_val, grd_global_val = \
                    sess.run([sat_global, grd_global], feed_dict=feed_dict)

                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]

            print('   compute accuracy')
            val_recall1, val_recall1percent = validate(grd_global_descriptor, sat_global_descriptor)
            print('   %d: Recall@1 = %.1f%%, Recall@1%% = %.1f%%' %
                (epoch, val_recall1 * 100.0, val_recall1percent * 100.0))
            with open('../Result/' + data_type + '/' + str(network_type) + '_accuracy.txt', 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + 
                        ' : Recall@1 = ' + str(val_recall1) + 
                        ' Recall@1% = ' + str(val_recall1percent) + '\n')


if __name__ == '__main__':
    train(start_epoch)