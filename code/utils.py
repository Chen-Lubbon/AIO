import random
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

import kornia

# 1. 感知损失：让图片看起来更“真”，而不是更“模糊”
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 VGG19 的特征层
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        # x: 恢复图, y: GT
        return F.mse_loss(self.vgg(x), self.vgg(y))

# 2. TV Loss：防止 T 和 M 出现乱七八糟的噪点，强制平滑
class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def global_color_loss(pred, gt):
    pred_mean = pred.mean(dim=[2,3])
    gt_mean = gt.mean(dim=[2,3])
    return F.l1_loss(pred_mean, gt_mean)

def ambient_color_prior(A, I):
    # I 的平均颜色
    mean_I = I.mean(dim=[2,3], keepdim=True)
    return F.l1_loss(A, mean_I)

def image_color_loss(J, label_I):
    eps = 1e-5
    J_safe = torch.clamp(J, eps, 1.0 - eps)
    GT_safe = torch.clamp(label_I, eps, 1.0 - eps)
    J_hsv  = kornia.color.rgb_to_hsv(J_safe)
    GT_hsv  = kornia.color.rgb_to_hsv(GT_safe)
    # 将 Hue 转换为弧度
    # H_pred_rad = J_hsv[:, 0:1, :, :] * 2 * 3.1415926535897
    # H_gt_rad = GT_hsv[:, 0:1, :, :] * 2 * 3.1415926535897
    # loss_hue = torch.mean(1 - torch.cos(H_pred_rad - H_gt_rad))
    # 伪代码，可使用 kornia.color.rgb_to_lab
    J_lab = kornia.color.rgb_to_lab(J_safe)
    GT_lab = kornia.color.rgb_to_lab(GT_safe)
    loss_lab = F.l1_loss(J_lab[:, 1:, :, :], GT_lab[:, 1:, :, :]) 
    # 增加色度通道的权重
    # return loss_hue + 0.5 * loss_lab
    return loss_lab

class LossManager:

    def __init__(self, model,
                 criterion_l1,
                 criterion_mse,
                 criterion_perceptual,
                 criterion_tv,
                 LossWts,
                 trans_epoch=30):

        self.model = model

        self.l1 = criterion_l1
        self.mse = criterion_mse
        self.perceptual = criterion_perceptual
        self.tv = criterion_tv

        self.w = LossWts

        self.trans_epoch = trans_epoch
        self.refine_frozen = False

    # -----------------------------
    # freeze / unfreeze refine
    # -----------------------------

    def freeze_module(self, model_name):
        for p in getattr(self.model, model_name).parameters():
            p.requires_grad = False
        setattr(self, f"{model_name}_frozen", True)

    def unfreeze_module(self, model_name):
        for p in getattr(self.model, model_name).parameters():
            p.requires_grad = True
        setattr(self, f"{model_name}_frozen", False)

    # -----------------------------
    # schedule
    # -----------------------------
    def update_stage(self, epoch):

        if epoch < self.trans_epoch and not self.refine_frozen:
            self.freeze_module('refine')

        if epoch >= self.trans_epoch and self.refine_frozen:
            print(f"Epoch {epoch}: Transitioning to Stage B - Unfreezing RefineNet")
            self.unfreeze_module('refine')

    # -----------------------------
    # compute loss
    # -----------------------------
    def compute(self,
                epoch,
                J, J0,
                T, M, A,
                degraded_I,
                label_I,
                label_T,
                label_M,
                label_A
                ):

        self.update_stage(epoch)

        # ------------------
        # physics parameter losses
        # ------------------
        loss_T = self.mse(T, label_T) + 0.01 * self.tv(T)

        loss_M = self.l1(M, label_M) + 0.01 * self.tv(M)

        loss_A = self.mse(A, label_A.view(-1, 3, 1, 1)) #+ 0.05 * self.l1(A, degraded_I.mean(dim=[2,3]).view(-1, 3, 1, 1))
        
        # -----------------
        # physics loss
        # -----------------
        loss_J_vgg = self.perceptual(J0, label_I)

        loss_J0 = self.l1(J0, label_I) + 0.1 * loss_J_vgg #+ 0.05 * global_color_loss(J0, label_I) + 0.1 * image_color_loss(J0, label_I)

        I_recon = J0 * T + M + (1 - T) * A
        
        loss_phys = self.l1(I_recon, degraded_I)
        # -----------------
        # Stage A Physical parameter train
        # -----------------
        if epoch < self.trans_epoch:

            total_loss = (
                self.w[2] * loss_T +
                self.w[3] * loss_M +
                self.w[4] * loss_A +
                self.w[1] * loss_J0 +
                self.w[0] * loss_phys
            )
        
        # -----------------
        # Stage B Final out train
        # -----------------
        else:
            
            loss_J_l1 = self.l1(J, label_I)

            loss_J_vgg = self.perceptual(J, label_I)

            # loss_J_color = image_color_loss(J, label_I)
            
            # loss_global_color = global_color_loss(J0, label_I)

            # # decay physics weight
            alpha = max(0.1, 1 - epoch / 100)

            total_loss = (
                alpha * (self.w[2] * loss_T + self.w[3] * loss_M + self.w[4] * loss_A + self.w[1] * loss_J0 + loss_phys)
                +
                (1 - alpha) * self.w[0] * (loss_J_l1 + 0.1 * loss_J_vgg )#+ 0.05 * loss_J_color + 0.1 * loss_global_color)
            )

        return total_loss
    
class DegredationSimulator:
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def _add_gaussian_noise(self, image, t_val):
        """
        根据透射率 T 动态添加噪声。T 越小（雾越浓），噪声相对越大。
        """
        sigma = 0.01 / (t_val + 0.1) # 简单的噪声增益函数
        noise = np.random.normal(0, sigma, image.shape)
        noisy_img = image + noise
        return np.clip(noisy_img, 0, 1)

    def _depth_blur(self, I, T, A):
        h, w, _ = I.shape
        
        J = I * T + A * (1 - T)
        # 3️⃣ 加入低频雾团（Perlin-like）
        noise = cv2.GaussianBlur(
            np.random.rand(h, w),
            (151,151),
            0
        )
        noise = (noise - noise.min())/(noise.max()-noise.min())
        fog_map = noise[...,None]

        J = J * (1 - 0.3*fog_map) + A * 0.3*fog_map

        # 4️⃣ 深度相关模糊（前向散射）
        # ksize = int(h/200)*2+1
        # blur = cv2.GaussianBlur(J, (ksize,ksize), 0)
        # alpha = depth[...,None]
        # J = J*(1-alpha) + blur*alpha

        # 5️⃣ 降低局部对比度（在LAB）
        lab = cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)
        L = cv2.GaussianBlur(L, (15,15), 0)
        lab = cv2.merge([L,a,b])
        J = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)/255.0
        return J

    def _generate_brightness_map(self, I):
        brightness_map = 1 - I.mean(axis=2, keepdims=True)
        brightness_map = cv2.GaussianBlur(brightness_map, (51,51), 0)
        brightness_map = np.expand_dims(brightness_map, axis=2)
        return brightness_map
        
    def simulate_fog(self, I, depth, t_range, A):
        '''
        产生雾图
        '''
        h, w, _ = I.shape

        # 1️⃣ 深度图（指数衰减）
        beta = random.uniform(1.5, 3.0)
        T = np.exp(-beta * depth)
        T = np.stack([T]*3, axis=-1)
        T = ( T-T.min() + t_range[0] ) * (t_range[1] - t_range[0]) * random.uniform(0.8, 1)
        J = self._depth_blur(I, T, A)
        M = np.zeros_like(J)
        
        return np.clip(J,0,1), np.clip(T,0,1), M, A
        
    def simulate_rain(self, I, depth, t_range=(0.5, 0.9), A=0.8, mask_count=3000):
        '''
        产生雨图
        '''
        h, w, _ = I.shape
        
        rain_layer = np.zeros_like(I)
        mask_count = np.random.poisson(mask_count)
        
        init_angle = random.uniform(70, 110)
        for _ in range(mask_count):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)

            length = random.randint(10, 40)
            angle = init_angle + random.uniform(-5, 5)
            brightness = random.uniform(0.3, 0.6)
            thickness = random.randint(1,2)

            end_x = int(x + length*np.cos(np.radians(angle)))
            end_y = int(y + length*np.sin(np.radians(angle)))

            cv2.line(
                rain_layer,
                (x,y),
                (end_x,end_y),
                (brightness,brightness,brightness),
                thickness
            )

        # 运动模糊
        ksize = 15
        kernel = np.zeros((ksize,ksize))
        kernel[:,ksize//2] = 1
        kernel = kernel/ksize
        rain_layer = cv2.filter2D(rain_layer,-1,kernel)

        # 轻微模糊
        rain_layer = cv2.GaussianBlur(rain_layer,(3,3),0)
        
        # 半透明叠加
        # 透射率
        beta = random.uniform(1.5, 3.0)
        T = np.exp(-beta * depth)
        T = np.stack([T]*3, axis=-1)
        T = T * (t_range[0] + (t_range[1] - t_range[0]) * random.uniform(0.8, 1))
        
        J = self._depth_blur(I, T, A)
        
        rain_layer = rain_layer * 0.5 * self._generate_brightness_map(I)
        J = J * (1 - rain_layer) + rain_layer

        return np.clip(J,0,1), np.clip(T,0,1), rain_layer, A

    def simulate_snow(self, I, depth, t_range=(0.5, 0.9), A=0.85, mask_count=2000):
        """
        产生雪图
        """
        h, w, _ = I.shape
        
        snow_layer = np.zeros_like(I)
        mask_count = np.random.poisson(mask_count)
        
        for _ in range(mask_count):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            
            size = random.randint(1, 3)
            
            opacity = random.uniform(0.3, 0.8)
            
            cv2.circle(
                snow_layer,
                (x, y),
                size,
                (opacity, opacity, opacity),
                -1
            )
        
        # 加运动模糊（下落）
        ksize = random.choice([1,2,3,4])
        kernel = np.zeros((ksize, ksize))
        kernel[:, ksize//2] = 1
        kernel = kernel / ksize
        
        snow_layer = cv2.filter2D(snow_layer, -1, kernel)
        
        # 模糊制造景深
        snow_layer = cv2.GaussianBlur(snow_layer, (5,5), 0)
        
        # 透射率
        beta = random.uniform(1.5, 3.0)
        T = np.exp(-beta * depth)
        T = np.stack([T]*3, axis=-1)
        T = T * (t_range[0] + (t_range[1] - t_range[0]) * random.uniform(0.8, 1))
        
        J = self._depth_blur(I, T, A)
        brightness_map = self._generate_brightness_map(J)
        snow_layer = snow_layer * 0.8 *brightness_map
        J = J * (1 - snow_layer) + snow_layer
        
        return np.clip(J, 0, 1), np.clip(T, 0, 1), snow_layer, A

    def simulate_underwater(self, I, depth, t_range=(0.2, 0.6), A=[0.8,0.8,0.8], water_type='blue'):
        '''
        产生水下图
        '''
        h, w = I.shape[:2]

        # 1️⃣ 物理修正的衰减系数 (Beta)
        # 比例参考：红光吸收最强，蓝/绿视水质而定
        beta = random.uniform(1.5, 3.0)
        water_params = {
            "blue": {
            "beta": (np.array([1.5, 0.2, 0.1], dtype=np.float32) * beta).tolist(),
            "A": [0.05, 0.3, 0.6]
            },
            "green": {
            "beta": (np.array([1.5, 0.1, 0.5], dtype=np.float32) * beta).tolist(),
            "A": [0.1, 0.6, 0.2]
            },
            "yellow": {
            "beta": (np.array([1.2, 0.8, 1.5], dtype=np.float32) * beta).tolist(),
            "A": [0.5, 0.4, 0.1]
            }
        }
        
        p = water_params[water_type]
        
        # 注入随机扰动
        beta = np.array(p["beta"]) * random.uniform(0.8, 1.2)
        A = np.array(p["A"]) * random.uniform(0.7, 1.0)

        # 2️⃣ 计算透射率 T
        T_r = np.exp(-beta[0] * depth)
        T_g = np.exp(-beta[1] * depth)
        T_b = np.exp(-beta[2] * depth)
        T = np.stack([T_r, T_g, T_b], axis=-1)

        T = T * random.uniform(*t_range)

        J = self._depth_blur(I, T, A)

        return J, T, np.zeros_like(I), A
    
    def degrade(self, I, depth, t_range=None , A=[0.7,0.7,0.7], mask_count=3000, method="fog"):
        A = np.array(A) * random.uniform(0.7, 1.0)
        if method == "fog":
            t_range = t_range if t_range else (0.3, 0.6)
            return self.simulate_fog(I, depth, t_range, A)
        elif method == "rain":
            t_range = t_range if t_range else (0.5, 0.9)
            return self.simulate_rain(I, depth, t_range, A, mask_count=mask_count)
        elif method == "snow":
            t_range = t_range if t_range else (0.5, 0.9)
            return self.simulate_snow(I, depth, t_range, A, mask_count=mask_count)
        elif method == "underwater1":
            t_range = t_range if t_range else (0.2, 0.6)
            return self.simulate_underwater(I, depth, t_range, A, water_type='blue')
        elif method == "underwater2":
            t_range = t_range if t_range else (0.2, 0.6)
            return self.simulate_underwater(I, depth, t_range, A, water_type='green')
        elif method == "underwater3":
            t_range = t_range if t_range else (0.2, 0.6)
            return self.simulate_underwater(I, depth, t_range, A, water_type='yellow')
        else:
            print(f"Unknown degradation method: {method}")
            return I, np.ones_like(depth), np.zeros_like(I), A
        
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # --- 使用示例 ---
    path_img = "data/landscape/2.jpg"
    path_depth = "data/depth/2.png"
    # 读取并归一化到 [0, 1]
    img = cv2.imread(path_img)
    depth = cv2.imread(path_depth) # 假设深度图已经归一化到 [0, 1]
    depth = depth[:,:,0] # 取单通道作为深度图
    h, w = img.shape[:2]
    h, w = h*3, w*3
    print(f"Original image size: {img.shape}, amped size: {h}x{w}")
    img = cv2.resize(img, (w, h))
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    depth = depth.astype(np.float32) / 255.0
    if img is None:
        raise FileNotFoundError("Image not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    simulator = DegredationSimulator(seed=4)
    # img_fogged = simulator.simulate_fog(img, depth, t_range=(0.2, 0.6), A=0.8)
    img_rained, _, _, _ = simulator.degrade(img, depth, t_range=(0.5, 1), A=[0.8,0.8,0.8], mask_count=5000, method="fog")
    img_snowed, _, _, _ = simulator.degrade(img, depth, t_range=(0.5, 1), A=0.8, mask_count=5000, method="snow")
    # img_underwater = simulator.simulate_underwater(img, depth, t_range=(0.2, 1), A=0.7)
    # plt.figure(figsize=(16, 4))
    # plt.subplot(1,5,1)
    # plt.imshow(img)
    # plt.title("Original Image")
    # plt.axis('off')
    # plt.subplot(1,5,2)
    # plt.imshow(img_fogged)
    # plt.title("Foggy Image")
    # plt.axis('off')
    # plt.subplot(1,5,3)
    # plt.imshow(img_rained)
    # plt.title("Rainy Image")
    # plt.axis('off')
    # plt.subplot(1,5,4)
    # plt.imshow(img_snowed)
    # plt.title("Snowy Image")
    # plt.axis('off')
    # plt.subplot(1,5,5)
    # plt.imshow(img_underwater)
    # plt.title("Underwater Image")
    # plt.axis('off')
    # plt.show()
    plt.imshow(img_rained)
    plt.show()
    plt.figure()
    plt.imshow(img_snowed)
    plt.show()