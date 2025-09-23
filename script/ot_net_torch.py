import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

# ---- sinkhorn: TF와 1:1 동작 ----
@torch.no_grad()
def sinkhorn_torch(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    return_2d = False
    if log_alpha.dim() == 2:
        log_alpha = log_alpha.unsqueeze(0)  # (1,N,N)
        return_2d = True
    elif log_alpha.dim() != 3:
        raise ValueError("log_alpha must be 2D or 3D")

    z = log_alpha
    for _ in range(n_iters):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)  # col-normalize
        z = z - torch.logsumexp(z, dim=1, keepdim=True)  # row-normalize

    out = torch.exp(z)
    if return_2d:
        out = out.squeeze(0)
    return out


# ---- VGG16 conv5 출력 백본(프리트레인 X) ----
def build_vgg16_conv5(pretrained: bool = False) -> nn.Sequential:
    m = vgg16(pretrained=pretrained).features
    # conv5_3 까지 + 마지막 maxpool 포함(/32 다운샘플) → 원 TF와 통상 동일
    # torchvision VGG16 features는 이미 conv/pool 순서가 표준임
    return m


# ---- Xavier init / Truncated normal init ----
def init_xavier_(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def trunc_normal_(tensor: torch.Tensor, std: float = 0.005):
    # 간단한 truncated normal: 2σ 클리핑
    with torch.no_grad():
        tensor.normal_(0, std)
        tensor.clamp_(-2*std, 2*std)
    return tensor


class CVFTTorch(nn.Module):
    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout_rate = 1.0 - keep_prob

        # 두 분기 백본 (동일 구조, 가중치 독립)
        self.backbone_sat = build_vgg16_conv5(pretrained=False)
        self.backbone_grd = build_vgg16_conv5(pretrained=False)

        # 축소 conv (3x3, stride=2, out=64) : 원 코드의 sat_conv / grd_conv
        self.sat_reduce = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.grd_reduce = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1, bias=True)

        # OT 분기의 1x1 conv (채널→1), grd 브랜치에서만 사용
        self.ot1x1_grd = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # fc_layer 파라미터는 공간 크기(N=H*W)에 의존 → lazy init
        self.fc_weight = None  # shape: (N, N*N)
        self.fc_bias   = None  # shape: (N*N,)

        # 초기화
        self.apply(init_xavier_)  # conv류 glorot_uniform
        # 1x1 conv bias 0은 위에서 이미 처리됨

    def _lazy_init_fc(self, N: int, device):
        if (self.fc_weight is None) or (self.fc_weight.shape != (N, N*N)):
            self.fc_weight = nn.Parameter(torch.empty(N, N*N, device=device))
            self.fc_bias   = nn.Parameter(torch.zeros(N*N, device=device))
            trunc_normal_(self.fc_weight, std=0.005)

    def forward(self, x_sat: torch.Tensor, x_grd: torch.Tensor):
        """
        x_*: float tensor, shape [B,3,H,W], 전처리는 원본처럼 BGR-mean 뺀 값을 그대로 사용해도 OK
        """

        # 1) 백본 추출 (VGG16 conv5_3 출력, [B,512,H',W'])
        sat = self.backbone_sat(x_sat)
        grd = self.backbone_grd(x_grd)

        # 2) 3x3/stride2 축소 → [B,64, Hs,Ws], [B,64,Hg,Wg]
        sat = self.sat_reduce(sat)
        grd = self.grd_reduce(grd)

        # 3) grd 공간 크기를 sat에 맞춤 (TF bilinear 기본값과 동일)
        Hs, Ws = sat.shape[2], sat.shape[3]
        grd = F.interpolate(grd, size=(Hs, Ws), mode='bilinear', align_corners=False)

        # ---- OT module (grd branch) ----
        # 4) grd → 1x1 conv → [B,1,Hs,Ws] → fc_layer: [B, N]→[B,N,N]
        B, C64, H, W = grd.shape
        N = H * W

        # lazy init fc
        self._lazy_init_fc(N, device=grd.device)

        conv1x1 = self.ot1x1_grd(grd)                  # [B,1,H,W]
        inp = conv1x1.view(B, N)                       # [B,N]
        fc = inp @ self.fc_weight + self.fc_bias       # [B, N*N]
        fc = fc.view(B, N, N)                          # [B,N,N]

        # 5) Sinkhorn(-100 스케일 그대로)
        ot_mat = sinkhorn_torch(fc * (-100.0), n_iters=20)  # [B,N,N]

        # 6) apply_ot: einsum('bci,bio->bco'), 여기서 입력은 grd 특징 [B,64,H,W]
        grd_reshaped = grd.view(B, C64, N)             # [B,64,N]
        out = torch.einsum('bci,bio->bco', grd_reshaped, ot_mat)  # [B,64,N]
        grd_ot = out.view(B, C64, H, W)                # [B,64,H,W]

        # 7) sat는 그대로, grd는 OT 적용본 사용
        sat_feat = sat
        grd_feat = grd_ot

        # 8) flatten + L2 normalize (axis=1)
        sat_vec = sat_feat.view(B, -1)
        grd_vec = grd_feat.view(B, -1)
        sat_vec = F.normalize(sat_vec, p=2, dim=1)
        grd_vec = F.normalize(grd_vec, p=2, dim=1)

        return sat_vec, grd_vec

# def VGG_conv(x_sat, x_grd, keep_prob, trainable):
#     def conv_layer(x, kernel_dim, input_dim, output_dim, stride, trainable, activated,
#                    name='ot_conv', activation_function=tf.nn.relu):
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  # reuse=tf.AUTO_REUSE
#             weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
#                                      trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
#             bias = tf.get_variable(name='biases', shape=[output_dim],
#                                    trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

#             out = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME') + bias

#             if activated:
#                 out = activation_function(out)

#             return out

#     ############## VGG module #################

#     vgg_grd = VGG16()
#     grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
#     grd_vgg = conv_layer(grd_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
#                          activated=True, name='grd_conv')

#     vgg_sat = VGG16()
#     sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
#     sat_vgg = conv_layer(sat_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
#                              activated=True, name='sat_conv')

#     ############## resize #################
#     height, width, channel = sat_vgg.get_shape().as_list()[1:]

#     grd_vgg = tf.image.resize_bilinear(grd_vgg, [height, width])

#     ############## reshape #################
#     grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
#     grd_global = tf.reshape(grd_vgg, [-1, grd_height * grd_width * grd_channel])

#     sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
#     sat_global = tf.reshape(sat_vgg, [-1, sat_height * sat_width * sat_channel])

#     return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


# def VGG_gp(x_sat, x_grd, keep_prob, trainable):

    # ############## VGG module #################
    # vgg_grd = VGG16()
    # grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')

    # vgg_sat = VGG16()
    # sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')

    # ############## Global pooling #################
    # grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    # grd_global = tf.nn.max_pool(grd_vgg, [1, grd_height, grd_width, 1], [1, 1, 1, 1], padding='VALID')
    # grd_global = tf.reshape(grd_global, [-1, grd_channel])

    # sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    # sat_global = tf.nn.max_pool(sat_vgg, [1, sat_height, sat_width, 1], [1, 1, 1, 1], padding='VALID')
    # sat_global = tf.reshape(sat_global, [-1, sat_channel])

    # return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)