# create_resnet_initial_ckpt_with_slots.py
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from ot_net import CVFT_ResNet  # 네가 쓰는 그 함수 (훈련 코드와 동일 버전)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 아무거나 1장

def main():
    # === 훈련 코드와 "동일한" placeholder/스코프/인자 ===
    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = True

    # 동일한 모델 경로 (CVFT_ResNet 내부 variable_scope/레이어 이름이 훈련과 1:1 일치해야 함)
    sat_g, grd_g = CVFT_ResNet(sat_x, grd_x, keep_prob, is_training)

    dummy_var = tf.Variable(0, name='Variable', dtype=tf.int32, trainable=False)

    # 더미 loss (슬롯 생성을 위해 필요). 값은 의미 없음
    loss = tf.add_n([tf.nn.l2_loss(v)*0.0 for v in tf.trainable_variables()])

    # 훈련 코드와 "동일한" name_scope 로 Adam 생성해야
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(1e-5, 0.9, 0.999)
        train_op = optimizer.minimize(loss, global_step=global_step)  # ← 이때 슬롯 변수가 생성됨

    # 슬롯 포함 전체 저장용 Saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    out_dir = '../Model/Initial_model_resnet'
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'Initial_model.ckpt')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        # (선택) 한 번 실행해서 모든 var 확실히 materialize
        sess.run(train_op, feed_dict={
            sat_x: np.zeros((1,256,256,3), np.float32),
            grd_x: np.zeros((1,112,616,3), np.float32),
            keep_prob: 1.0
        })

        save_path = saver.save(sess, ckpt_path)
        print('Saved:', save_path)

if __name__ == '__main__':
    main()