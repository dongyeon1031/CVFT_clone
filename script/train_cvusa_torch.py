#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 너가 이미 만든 토치 버전 모듈들
from input_data_cvusa_torch import CVUSADataset
from ot_net_torch import CVFTTorch  # TF의 CVFT를 토치로 옮긴 클래스(앞서 만든 버전)

# ------------------ 하이퍼파라미터 / 인자 ------------------
parser = argparse.ArgumentParser(description='PyTorch CVFT training (CVUSA)')
parser.add_argument('--data_root', type=str, default='../Data/CVUSA', help='CVUSA root')
parser.add_argument('--splits_train', type=str, default='splits/train-19zl.csv')
parser.add_argument('--splits_val',   type=str, default='splits/val-19zl.csv')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0, help='decoupled wd 쓰면 optimizer에서 설정')
parser.add_argument('--loss_weight', type=float, default=10.0)
parser.add_argument('--hard_k', type=int, default=5, help='0이면 hard mining 안함, >0이면 top-k 사용')
parser.add_argument('--keep_prob', type=float, default=0.8)  # 드롭아웃 rate=1-keep_prob 로 사용
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_root', type=str, default='../Model/CVUSA/CVFT_torch')
parser.add_argument('--resume', type=str, default='', help='resume ckpt path (optional)')
parser.add_argument('--init_ckpt', type=str, default='../Model/Initial_model/Initial_model.ckpt', help='initial weights ckpt (optional, torch format)')
parser.add_argument('--gpu', type=str, default='0,1', help='e.g., "0" or "0,1"')
args = parser.parse_args()

print('cuda available:', torch.cuda.is_available())
print('torch cuda:', torch.version.cuda)
print('device count:', torch.cuda.device_count())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ 데이터 ------------------
# 주의: Dataset은 원 TF 전처리와 동일( OpenCV BGR + VGG mean subtraction )로 만들어둔 상태여야 함
train_set = CVUSADataset(
    img_root=args.data_root,
    csv_path=os.path.join(args.data_root, args.splits_train),
    resize_sat=(256, 256),
    resize_grd=(616, 112),
    return_id=False,  # 학습은 인덱스만 있으면 충분
)
val_set = CVUSADataset(
    img_root=args.data_root,
    csv_path=os.path.join(args.data_root, args.splits_val),
    resize_sat=(256, 256),
    resize_grd=(616, 112),
    return_id=False,
)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

# ------------------ 모델 ------------------
model = CVFTTorch(keep_prob=args.keep_prob)  # 내부에서 VGG16 conv → conv(3x3,stride2,64ch) → OT(grd) → flatten+L2 normalize
model = model.to(device)

# 멀티 GPU 쓰려면 (원하면 주석 해제)
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs via DataParallel')
    model = nn.DataParallel(model, device_ids=[0, 1])

# 초기 체크포인트 로드(토치 포맷일 때만!)
if args.init_ckpt and Path(args.init_ckpt).exists():
    print(f'Loading initial checkpoint: {args.init_ckpt}')
    s = torch.load(args.init_ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(s['model'], strict=False) if 'model' in s else model.load_state_dict(s, strict=False)
    print('Loaded init ckpt. missing:', missing, ' unexpected:', unexpected)

# ------------------ 손실(원 TF 수식 그대로) ------------------
def soft_margin_triplet_loss(sat, grd, batch_hard_count=0, loss_weight=10.0):
    """
    sat, grd: [B, D], L2-normalized
    dist = 2 - 2 * sat @ grd^T  (코사인 거리와 동치)
    """
    # [B, B]
    dist = 2.0 - 2.0 * (sat @ grd.t())
    pos = torch.diag(dist)  # [B]
    B = dist.size(0)

    if batch_hard_count == 0:
        # 전체 쌍 평균 (원본과 동일)
        g2s = pos.unsqueeze(1) - dist     # [B,B]
        s2g = pos.unsqueeze(0) - dist     # [B,B]
        loss_g2s = torch.log1p(torch.exp(g2s * loss_weight)).sum() / (B*(B-1))
        loss_s2g = torch.log1p(torch.exp(s2g * loss_weight)).sum() / (B*(B-1))
        return 0.5*(loss_g2s + loss_s2g)
    else:
        g2s = torch.log1p(torch.exp((pos.unsqueeze(1) - dist) * loss_weight)) # [B,B]
        s2g = torch.log1p(torch.exp((pos.unsqueeze(0) - dist) * loss_weight)) # [B,B]
        topk_g2s, _ = torch.topk(g2s.t(), batch_hard_count, dim=0)
        topk_s2g, _ = torch.topk(s2g, batch_hard_count, dim=1)
        return 0.5*(topk_g2s.mean() + topk_s2g.mean())

# ------------------ 옵티마이저/스케줄 ------------------
# Adam(기본), AdamW 쓰려면 optim.AdamW로 교체 + weight_decay 인자 사용
# (주의: AdamW의 WD는 decoupled WD)
use_adamw = False
if use_adamw and args.weight_decay > 0:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # WD는 loss에 L2 추가하거나 수동 디커플링으로 가능

# 스케줄러(선택)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ------------------ 유틸 ------------------
def validate_epoch(model, loader):
    model.eval()
    sat_all = []
    grd_all = []
    with torch.no_grad():
        for sat, grd, _ in loader:
            sat = sat.to(device, non_blocking=True).float()
            grd = grd.to(device, non_blocking=True).float()
            sat_g, grd_g = model(sat, grd)  # [B, D], [B, D]
            sat_all.append(sat_g.detach().cpu().numpy())
            grd_all.append(grd_g.detach().cpu().numpy())
    sat_all = np.concatenate(sat_all, axis=0)
    grd_all = np.concatenate(grd_all, axis=0)

    # TF validate와 동일: Recall@1 / Recall@1%
    dist = 2 - 2 * (sat_all @ grd_all.T)
    N = dist.shape[0]
    top1p = int(N * 0.01) + 1

    correct_r1 = 0
    correct_r1p = 0
    for i in range(N):
        gt = dist[i, i]
        rank = np.sum(dist[:, i] < gt)  # 열 기준
        if rank == 0:
            correct_r1 += 1
        if rank < top1p:
            correct_r1p += 1

    return correct_r1 / N, correct_r1p / N

# ------------------ 학습 루프 ------------------
start_epoch = 0
if args.resume and Path(args.resume).exists():
    print(f'Resuming from: {args.resume}')
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    start_epoch = ckpt.get('epoch', 0) + 1
    print('Resumed at epoch', start_epoch)

save_root = Path(args.save_root)
save_root.mkdir(parents=True, exist_ok=True)

for epoch in range(start_epoch, args.epochs):
    model.train()
    running = 0.0
    for it, (sat, grd, _) in enumerate(train_loader):
        sat = sat.to(device, non_blocking=True).float()
        grd = grd.to(device, non_blocking=True).float()

        sat_g, grd_g = model(sat, grd)  # [B,D] L2-normalized
        loss = soft_margin_triplet_loss(
            sat_g, grd_g,
            batch_hard_count=args.hard_k,
            loss_weight=args.loss_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        if (it % 20) == 0:
            avg = running / max(1, (it+1))
            print(f'epoch {epoch} iter {it}: loss {avg:.4f}')

    # if scheduler is not None:
    #     scheduler.step()

    # 저장
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': vars(args),
    }
    torch.save(ckpt, str(save_root / f'epoch_{epoch}.pth'))
    print(f'[Saved] {save_root}/epoch_{epoch}.pth')

    # 검증
    r1, r1p = validate_epoch(model, val_loader)
    print(f'Validate: epoch {epoch}: Recall@1={r1*100:.1f}%, Recall@1%={r1p*100:.1f}%')

    # 결과 로그 파일(TF와 유사)
    result_dir = Path('../Result/CVUSA')
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / 'CVFT_torch_accuracy.txt', 'a') as f:
        f.write(f'{epoch} : Recall@1={r1:.6f} Recall@1%={r1p:.6f}\n')