import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def make_cvusa_loaders(img_root="../Data/CVUSA",
                       train_csv="splits/train-19zl.csv",
                       val_csv="splits/val-19zl.csv",
                       batch_size=32, num_workers=4, pin_memory=True):
    train_ds = CVUSADataset(os.path.join(img_root, train_csv), img_root, split="train")
    val_ds   = CVUSADataset(os.path.join(img_root, val_csv),   img_root, split="val", return_id=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    # 검증은 순서를 유지해야 Recall 평가가 깔끔함
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


class CVUSADataset(Dataset):
    """
    CSV 한 줄: sat_rel_path,grd_rel_path,....
    반환: (sat_tensor, grd_tensor, pano_id or idx)
    전처리: VGG 스타일 BGR 평균감산 유지 (기존 코드와 동일)
    """
    def __init__(self, csv_path, img_root, split="train",
                 resize_sat=(256, 256), resize_grd=(616, 112),
                 return_id=False):
        self.img_root = img_root
        self.resize_sat = resize_sat
        self.resize_grd = resize_grd
        self.return_id = return_id

        self.items = []
        with open(csv_path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                sat_rel = data[0]
                grd_rel = data[1]
                pano_id = os.path.splitext(os.path.basename(data[0]))[0]
                self.items.append((sat_rel, grd_rel, pano_id))

        print(f"[CVUSADataset] loaded {len(self.items)} items from {csv_path} (split={split})")

        # VGG BGR mean (기존 코드 값)
        self.bgr_mean = np.array([103.939, 116.779, 123.6], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def _load_and_preproc(self, full_path, size_hw, kind):
        """
        full_path: 이미지 전체 경로
        size_hw: (W,H) 순 아닌 점 주의 — cv2.resize는 (W,H)
        kind: 'sat' 또는 'grd' (디버그 메시지용)
        """
        img = cv2.imread(full_path)  # BGR, uint8, HxWxC
        if img is None:
            raise FileNotFoundError(f"fail to read: {full_path}")

        # 원본 체크 로직을 유지하고 싶다면(디버그 용)
        if kind == 'sat':
            # 원래 코드: 정사각 확인
            if img.shape[0] != img.shape[1]:
                # 경고만 띄우고 resize는 진행
                print(f"[warn] sat not square: {full_path}, shape={img.shape}")
        else:  # 'grd'
            # 원래 코드: (224, 1232) 체크
            if not (img.shape[0] == 224 and img.shape[1] == 1232):
                print(f"[warn] grd unusual shape: {full_path}, shape={img.shape}")

        img = cv2.resize(img, size_hw, interpolation=cv2.INTER_AREA)  # (W,H)
        img = img.astype(np.float32)

        # VGG BGR mean subtraction (기존과 동일)
        img[:, :, 0] -= self.bgr_mean[0]  # B
        img[:, :, 1] -= self.bgr_mean[1]  # G
        img[:, :, 2] -= self.bgr_mean[2]  # R

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        return torch.from_numpy(img)  # float32 tensor

    def __getitem__(self, idx):
        sat_rel, grd_rel, pano_id = self.items[idx]
        sat = self._load_and_preproc(os.path.join(self.img_root, sat_rel),
                                     self.resize_sat, kind='sat')
        grd = self._load_and_preproc(os.path.join(self.img_root, grd_rel),
                                     self.resize_grd, kind='grd')
        if self.return_id:
            return sat, grd, pano_id
        else:
            return sat, grd, idx