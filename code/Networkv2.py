import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 基础残差块：比原本的 ConvBlock 更深且易于收敛
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride) if in_c != out_c or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        return F.leaky_relu(self.conv(x) + self.shortcut(x), 0.2)

# -----------------------------
# 增强型 Shared Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样 4 次，感受野足以覆盖 960x540 的全局信息
        self.enc1 = ResBlock(3, 32)   # 1/1
        self.enc2 = ResBlock(32, 64, stride=2)  # 1/2
        self.enc3 = ResBlock(64, 128, stride=2) # 1/4
        self.enc4 = ResBlock(128, 256, stride=2)# 1/8
        self.enc5 = ResBlock(256, 512, stride=2)# 1/16

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return [f1, f2, f3, f4, f5] # 返回列表以支持 Skip Connection

# -----------------------------
# 增强型 Transmission Decoder (带 Skip)
# -----------------------------
class TDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 ConvTranspose2d 配合 ResBlock 恢复细节
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ResBlock(512, 256) # 256(up) + 256(skip)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResBlock(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ResBlock(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec4 = ResBlock(64, 32)
        
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        
        x = self.up1(f5)
        x = self.dec1(torch.cat([x, f4], dim=1))
        
        x = self.up2(x)
        x = self.dec2(torch.cat([x, f3], dim=1))
        
        x = self.up3(x)
        x = self.dec3(torch.cat([x, f2], dim=1))
        
        x = self.up4(x)
        x = self.dec4(torch.cat([x, f1], dim=1))
        
        return torch.sigmoid(self.out(x))

# -----------------------------
# 增强型 Mask Decoder (结构与 T 类似但参数独立)
# -----------------------------
class MDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ResBlock(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec4 = ResBlock(64, 32)
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        x = self.dec1(torch.cat([self.up1(f5), f4], dim=1))
        x = self.dec2(torch.cat([self.up2(x), f3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), f2], dim=1))
        x = self.dec4(torch.cat([self.up4(x), f1], dim=1))
        return self.out(x)

# -----------------------------
# 大气光 Head (加深 MLP)
# -----------------------------
class AHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, features):
        f5 = features[-1]
        x = self.avg_pool(f5)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1, 3, 1, 1)

# -----------------------------
# RefineNet (增加残差结构)
# -----------------------------
class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv2d(6, 64, 3, padding=1)
        self.res_layers = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, I, J0):
        x = torch.cat([I, J0], dim=1)
        x = self.init_conv(x)
        x = self.res_layers(x)
        return J0 + self.out(x)

# -----------------------------
# 主模型 (处理尺寸对齐)
# -----------------------------
class UnifiedRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "UnifiedRestorationNet_v2"
        self.encoder = Encoder()
        self.t_decoder = TDecoder()
        self.m_decoder = MDecoder()
        self.a_head = AHead()
        self.refine = RefineNet()

    def forward(self, I):
        # --- 尺寸对齐 (Padding) ---
        h, w = I.shape[2], I.shape[3]
        # 4次下采样意味着需要被 2^4=16 整除
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        I_pad = F.pad(I, (0, pad_w, 0, pad_h), mode='reflect')

        # --- Forward ---
        feats = self.encoder(I_pad)
        T_pad = self.t_decoder(feats)
        M_pad = self.m_decoder(feats)
        A = self.a_head(feats)

        # --- 裁剪回原尺寸 ---
        T = T_pad[:, :, :h, :w]
        M = M_pad[:, :, :h, :w]

        # --- 物理层 (直接写在 forward 里更灵活) ---
        T_clamp = torch.clamp(T, 0.1, 1.0)
        J0 = (I - M - (1 - T_clamp) * A) / (T_clamp + 1e-3)

        # --- 精炼 ---
        J = self.refine(I, J0)

        return J, J0, T, M, A

if __name__ == '__main__':
    # 测试 540x960 尺寸
    model = UnifiedRestorationNet().cuda()
    input_img = torch.randn(1, 3, 540, 960).cuda()
    J, J0, T, M, A = model(input_img)
    print(f"输入: {input_img.shape}")
    print(f"输出 J: {J.shape}, T: {T.shape}, A: {A.shape}")