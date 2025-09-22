#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ------------------------
# 재현성(선택)
# ------------------------
tf.set_random_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 사용 GPU 지정(원하면 주석 처리)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ⬇️ 반드시 ot_net.py에 'CVFT_ResNet' (혹은 동일한 네이밍) 함수가 있어야 함
#    그리고 학습 코드에서 호출하는 스코프/레이어 이름과 100% 동일한 경로로 변수를 생성해야 함!
from ot_net import CVFT_ResNet


def dump_tensor_names_to_file(tensors, path):
    with open(path, "w") as f:
        for t in tensors:
            f.write(t.name + "\n")


def main():
    # ------------------------
    # 입력 placeholder: 학습 코드와 완전히 동일해야 함
    # ------------------------
    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # 학습 그래프와 같은 조건(BN 등)
    is_training = True

    # ------------------------
    # 그래프 빌드 (학습에서 쓰는 ResNet 경로와 스코프/이름 완벽히 동일!)
    # ------------------------
    sat_global, grd_global = CVFT_ResNet(sat_x, grd_x, keep_prob, is_training)

    # 옵티마이저/훈련연산 만들지 마세요! (슬롯 변수 방지)
    vars_to_save = tf.global_variables()

    print("[INFO] total variables to save:", len(vars_to_save))
    for v in vars_to_save[:10]:
        print("  EXAMPLE VAR:", v.name)
    # 전체 변수 이름 덤프(그래프 기준)
    os.makedirs("./_init_ckpt_debug", exist_ok=True)
    dump_tensor_names_to_file(vars_to_save, "./_init_ckpt_debug/graph_vars_resnet.txt")

    # Saver는 모델 변수만(지금은 옵티마이저 없으니 global_variables로 충분)
    saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=None)

    # 출력 경로
    out_dir = '../Model/Initial_model_resnet'
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'Initial_model.ckpt')

    # ------------------------
    # 세션 실행 & 저장
    # ------------------------
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        save_path = saver.save(sess, ckpt_path)
        print("[OK] ResNet 초기 체크포인트 저장:", save_path)

        # 저장된 ckpt의 변수 목록도 덤프(이름/개수 확인용)
        try:
            from tensorflow.python.training import py_checkpoint_reader
            reader = py_checkpoint_reader.NewCheckpointReader(save_path)
            ckpt_vars = reader.get_variable_to_shape_map()
            print("[INFO] variables in ckpt:", len(ckpt_vars))
            with open("./_init_ckpt_debug/ckpt_vars_resnet.txt", "w") as f:
                for k in sorted(ckpt_vars.keys()):
                    f.write(k + "\n")
            print("[OK] 변수 이름 덤프 완료: _init_ckpt_debug/ckpt_vars_resnet.txt")
        except Exception as e:
            print("[WARN] ckpt 변수 읽기 중 경고:", repr(e))

    print("\n[NEXT]")
    print(" - 학습 코드에서 start_epoch == 0 이고, ResNet 백본을 쓸 때는")
    print("   load_model_path = '../Model/Initial_model_resnet/Initial_model.ckpt' 로 설정하세요.")
    print(" - 로드 실패 시, _init_ckpt_debug/graph_vars_resnet.txt 와 ckpt_vars_resnet.txt 를 비교해")
    print("   누락/이름불일치 변수를 확인하세요.")


if __name__ == "__main__":
    main()