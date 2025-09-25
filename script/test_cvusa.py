#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:02:57 2019

@author: yujiao
"""

from input_data_cvusa import InputData
from ot_net import *

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os

import matplotlib.pyplot as plt

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='CVFT')

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type

data_type = 'CVUSA'

batch_size = 32

# -------------------------------------------------------- #

# --------------------- Retrieval Visualization ---------------------
# ===== t-SNE 시각화 & 임베딩 저장 =====
def save_tsne_plot(sat_feat, grd_feat, out_path="./tsne_embeddings.png",
                   max_points=5000, seed=42, normalize=True):
    """
    sat_feat, grd_feat: np.ndarray [N, D] 형태의 위성/지상 임베딩, huh
    normalize=True면 L2 정규화 후 투영, what
    """
    import numpy as np
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[WARN] scikit-learn이 없어 t-SNE를 건너뜀: pip install scikit-learn", flush=True)
        return

    S = sat_feat.copy()
    G = grd_feat.copy()
    if normalize:
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

    # 샘플링
    rng = np.random.RandomState(seed)
    ns = min(len(S), max_points // 2)
    ng = min(len(G), max_points - ns)
    idx_s = rng.choice(len(S), size=ns, replace=False)
    idx_g = rng.choice(len(G), size=ng, replace=False)

    X = np.vstack([G[idx_g], S[idx_s]])
    Y = np.array([0]*ng + [1]*ns)  # 0=query(grd), 1=reference(sat)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=seed)
    X2 = tsne.fit_transform(X)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    plt.scatter(X2[Y==0,0], X2[Y==0,1], s=6, alpha=0.6, label="Query (grd)")
    plt.scatter(X2[Y==1,0], X2[Y==1,1], s=6, alpha=0.6, label="Reference (sat)")
    plt.legend()
    plt.title("t-SNE of Embeddings (sampled)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[INFO] Saved t-SNE plot to {out_path}", flush=True)

def _resolve_paths(row, img_root="../Data/CVUSA"):
    """
    row: list/tuple 또는 문자열.
      - 보통 ['bingmap/19/xxx.jpg','streetview/panos/xxx.jpg','xxx'] (3칸)
      - 드물게 2칸일 수도 있음.
    return: (street_abs_path, bing_abs_path, sample_id or None)
    """
    # row가 문자열로 올 경우(혹시모를 케이스) 콤마 기준 분리 시도
    if isinstance(row, str):
        parts = [p.strip() for p in row.split(',') if p.strip()]
    elif isinstance(row, (list, tuple)):
        parts = list(row)
    else:
        raise ValueError(f"Unexpected row type: {type(row)}")

    if len(parts) < 2:
        raise ValueError(f"Row has <2 fields: {parts}")

    # 표준: 0=bing, 1=street, 2=id(옵션)
    bing_rel   = parts[0]
    street_rel = parts[1]
    sample_id  = parts[2] if len(parts) > 2 else None

    # 절대 경로로 변환
    bing_abs   = os.path.join(img_root, bing_rel)
    street_abs = os.path.join(img_root, street_rel)
    return street_abs, bing_abs, sample_id

def _topk_for_query(dist_array, q_idx, k=5):
    """열 기준(ground q_idx)로 거리가 작은 순서 top-k 위성 인덱스 반환"""
    column = dist_array[:, q_idx]  # shape [N]
    order = np.argsort(column)     # 작은 거리(유사) 우선
    return order[:k].tolist()

def _draw_border(ax, color="g", linewidth=4):
    """축에 컬러 테두리 그리기"""
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)

def visualize_random_retrieval(dist_array, input_data, top_k=5, num_queries=5, seed=2024,
                               fixed_indices=None, img_root="../Data/CVUSA"):
    """
    dist_array: [N, N] (sat vs grd) ; (i,i)가 GT
    input_data: InputData 인스턴스 (get_test_list 사용)
    top_k: 각 쿼리에 대해 보여줄 위성 이미지 개수
    num_queries: 시각화할 쿼리 개수
    seed: 랜덤 시드 (fixed_indices 없을 때 사용)
    fixed_indices: 특정 쿼리 인덱스 리스트(옵션)
    img_root: 이미지 루트 디렉토리 (예: ../Data/CVUSA)
    """
    # 1) 테스트 리스트 확보
    test_list = input_data.get_test_list()
    N = len(test_list)
    if N == 0:
        print("[WARN] test_list is empty.")
        return

    # 2) 쿼리 인덱스 선택
    if fixed_indices is not None and len(fixed_indices) > 0:
        query_indices = [i for i in fixed_indices if 0 <= i < N][:num_queries]
    else:
        rng = np.random.RandomState(seed)
        query_indices = rng.choice(N, size=min(num_queries, N), replace=False).tolist()

    # 3) 각 쿼리에 대해 시각화
    for qi in query_indices:
        row = test_list[qi]
        # 쿼리용 경로(Streetview), GT 위성 경로
        try:
            q_street_path, _, qi_id = _resolve_paths(test_list[qi], img_root)
        except Exception as e:
            print(f"[WARN] resolve_paths failed for query index {qi}: {e}")
            continue

        # top-k 후보 인덱스
        top_idxs = _topk_for_query(dist_array, qi, k=top_k)

        # 4) 그림 그리드: 1행(쿼리 2장: street(좌), GT bing(우)), 2행(top-k bing)
        fig = plt.figure(figsize=(2*(top_k+1), 6))
        fig.suptitle(f"Query #{qi_id} — top-{top_k} retrieval", fontsize=14)

        # (1) 1행 왼쪽: 쿼리 Streetview
        ax_q = plt.subplot(2, top_k+1, 1)  # 2행, (top_k+1)열 그리드 중 첫 칸
        try:
            img_q = plt.imread(q_street_path)
            ax_q.imshow(img_q)
            ax_q.set_title("Query (Streetview)")
        except Exception as e:
            ax_q.text(0.5, 0.5, f"Load fail:\n{os.path.basename(q_street_path)}",
                      ha='center', va='center')
            print(f"[WARN] 쿼리 이미지 로드 실패: {q_street_path} ({e})")
        ax_q.axis("off")
        _draw_border(ax_q, color="g", linewidth=4)  # 쿼리 자체는 녹색 테두리

        # (2) 1행 오른쪽: 정답 GT Bing (시각적으로 비교용)
        ax_gt = plt.subplot(2, top_k+1, top_k+1)  # 1행의 마지막 칸
        try:
            _, gt_bing, _ = _resolve_paths(test_list[qi], img_root)
            img_gt = plt.imread(gt_bing)
            ax_gt.imshow(img_gt)
            ax_gt.set_title("GT (Bing)")
        except Exception as e:
            ax_gt.text(0.5, 0.5, f"Load fail:\n{os.path.basename(gt_bing)}",
                       ha='center', va='center')
            print(f"[WARN] GT 이미지 로드 실패: {gt_bing} ({e})")
        ax_gt.axis("off")
        _draw_border(ax_gt, color="g", linewidth=4)  # GT는 녹색 테두리

        # (3) 2행: top-k Retrieval 결과 (Bing)
        for rank, si in enumerate(top_idxs, start=1):
            ax = plt.subplot(2, top_k+1, (top_k+1) + rank)  # 2행에서 rank 위치
            try:
                _, cand_bing, si_id = _resolve_paths(test_list[si], img_root)
                img_cand = plt.imread(cand_bing)
                ax.imshow(img_cand)
                is_correct = (si == qi)
                ax.set_title(f"#{rank} idx={si_id}",
                     color=("g" if is_correct else "r"))  # 제목 색상도 맞춤
                _draw_border(ax, color=("g" if is_correct else "r"), linewidth=4)
            except Exception as e:
                ax.text(0.5, 0.5, f"Load fail:\nidx={si}",
                        ha='center', va='center')
                print(f"[WARN] 후보 이미지 로드 실패: {cand_bing} ({e})")
                _draw_border(ax, color="r", linewidth=4)
            ax.axis("off")

        plt.tight_layout()

        # 저장 (원하면 경로 바꾸기)
        save_path = f"../Visualization/query{qi}.png"
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved visualization to {save_path}") 
# -------------------------------------------------------------------

def eval_soft_margin_triplet_loss(dist_array, loss_weight=10.0, batch_hard_k=0):
    """
    dist_array: [N, N], (i,i)가 정답쌍 거리. 작을수록 유사.
    훈련 코드의 compute_loss와 동일한 형태로 평가용 loss를 계산.
    """
    # sat x grd 기준
    pos = np.diag(dist_array)                          # [N]
    # g2s
    trip_g2s = pos[None, :] - dist_array              # [N, N]
    loss_g2s = np.log1p(np.exp(trip_g2s * loss_weight))
    # s2g
    trip_s2g = pos[:, None] - dist_array              # [N, N]
    loss_s2g = np.log1p(np.exp(trip_s2g * loss_weight))

    if batch_hard_k and batch_hard_k > 0:
        # in-batch hard mining 흉내 (평가에도 참고용으로만)
        topk_g2s = np.partition(loss_g2s.T, -batch_hard_k, axis=0)[-batch_hard_k:, :].mean()
        topk_s2g = np.partition(loss_s2g, -batch_hard_k, axis=1)[:, -batch_hard_k:].mean()
        return 0.5 * (topk_g2s + topk_s2g)
    else:
        N = dist_array.shape[0]
        pair_n = N * (N - 1.0)
        loss_g2s = (loss_g2s.sum() - np.log1p(np.exp(0)).sum()) / pair_n  # 대각 포함 보정은 생략 가능
        loss_s2g = (loss_s2g.sum() - np.log1p(np.exp(0)).sum()) / pair_n
        return 0.5 * (loss_g2s + loss_s2g)

def validate(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


if __name__ == '__main__':
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    tf.reset_default_graph()
    input_data = InputData()

    # define placeholders

    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')

    keep_prob = tf.placeholder(tf.float32)

    # build model
    if network_type == 'CVFT':
        sat_global, grd_global = CVFT(sat_x, grd_x, keep_prob, False)
    elif network_type == 'VGG_conv':
        sat_global, grd_global = VGG_conv(sat_x, grd_x, keep_prob, False)
    elif network_type == 'VGG_gp':
        sat_global, grd_global = VGG_gp(sat_x, grd_x, keep_prob, False)

    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')
        # load_model_path = '../Model/trained_model/CVUSA/CVFT/model.ckpt'
        load_model_path = "../Model/CVUSA/CVFT/base/40/model.ckpt"
        saver.restore(sess, load_model_path)


        # ---------------------- validation ----------------------

        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()

        val_i = 0
        total_size = input_data.get_test_dataset_size()
        while True:
            # print('      progress %d' % val_i)
            print(f'      progress {val_i}/{total_size} ({val_i/total_size:.2%})')
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
        dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(grd_global_descriptor))
        top1_percent = int(dist_array.shape[0] * 0.01) + 1
        val_accuracy = np.zeros((1, top1_percent))
        print('start')
        for i in range(top1_percent):
            val_accuracy[0, i] = validate(dist_array, i)

        print(network_type, ':')
        print('top1', ':', val_accuracy[0,1])
        print('top5', ':', val_accuracy[0,5])
        print('top10', ':', val_accuracy[0,10])
        print('top1%', ':', val_accuracy[0,-1])
        eval_loss = eval_soft_margin_triplet_loss(dist_array, loss_weight=10.0, batch_hard_k=0)
        print(f'eval soft-margin triplet loss: {eval_loss:.6f}')

        visualize_random_retrieval(dist_array, input_data, top_k=5, num_queries=5, seed=2024)
        # 임베딩 npy 저장 & t-SNE 호출
        np.save("./grd_feats.npy", grd_global_descriptor)
        np.save("./sat_feats.npy", sat_global_descriptor)
        save_tsne_plot(sat_global_descriptor, grd_global_descriptor,
                    out_path="./tsne_embeddings.png",
                    max_points=5000, seed=2025, normalize=True)
        # =====================================

        # ---------- Top-1 cosine similarity histogram ----------
        # (1) 코사인 유사도 행렬 S = sat_norm @ grd_norm^T
        sat_norm = sat_global_descriptor / (np.linalg.norm(sat_global_descriptor, axis=1, keepdims=True) + 1e-12)
        grd_norm = grd_global_descriptor / (np.linalg.norm(grd_global_descriptor, axis=1, keepdims=True) + 1e-12)
        S = np.matmul(sat_norm, grd_norm.T)  # shape [N_sat, N_grd]

        # (2) 각 쿼리(grd, 열)마다 Top-1 위성 인덱스와 점수
        best_i = np.argmax(S, axis=0)  # length N_grd
        top1_scores = S[best_i, np.arange(S.shape[1])]  # length N_grd
        correct_mask = (best_i == np.arange(S.shape[1]))  # GT는 대각선(i==j)

        # (3) 히스토그램 저장
        plt.figure(figsize=(8,6))
        plt.hist(top1_scores[correct_mask], bins=50, alpha=0.85, label="Top1 Correct")
        plt.hist(top1_scores[~correct_mask], bins=50, alpha=0.85, label="Top1 Incorrect")
        plt.title("Top-1 Similarity Distribution")
        plt.xlabel("cosine similarity")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./top1_similarity_hist.png", dpi=220)
        plt.close()
        print("[INFO] Saved ./top1_similarity_hist.png")
