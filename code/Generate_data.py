'''
    从data读取图像很深度，生成不同的恶劣天气图像和对应的maps，并保存到data文件夹
'''
    
import cv2
from utils import DegredationSimulator
import numpy as np
import os
from tqdm import tqdm
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

os.chdir(os.path.dirname(__file__))  # 切换到当前脚本所在目录，确保路径正
if __name__ == "__main__":
    
    path_img = "data/landscape"
    path_depth = "data/depth"
    os.makedirs("data/degraded", exist_ok=True)
    os.makedirs("data/transmission", exist_ok=True)
    os.makedirs("data/mask", exist_ok=True)
    os.makedirs("data/atmosphere", exist_ok=True)
    
    images = os.listdir(path_img)
    images = [img for img in images if img.endswith('.jpg') or img.endswith('. png')]
    images.sort(key=natural_sort_key)
    depths = os.listdir(path_depth)
    depths = [d for d in depths if d.endswith('.jpg') or d.endswith('.png')]
    depths.sort(key=natural_sort_key)  # 确保图像和深度图的顺序一致
    print(f"Found {len(images)} images in {path_img}.")
    print(f"Found {len(depths)} depth maps in {path_depth}.")
    change_interv = len(images) // 4
    print(f"Changing degradation type every {change_interv} images.")
    
    # 读取并归一化到 [0, 1]
    
    simulator = DegredationSimulator()
    methods = ["fog", "rain", "snow", "underwater"]
    method_index = 0
    print(f"Starting degradation with method: {methods[0]}")
    with tqdm(enumerate(zip(images, depths)), desc='Generating', total=len(images), mininterval=5) as pbar:
        for index, (img_file, depth_file) in pbar:
            # 计算当前应该使用的 method 索引
            # 使用 min() 防止 index // change_interv 溢出（当 index 是最后一个元素且不能整除时）
            method_index = min(index // change_interv, len(methods) - 1)
            current_method = methods[method_index]
            
            # 更新进度条显示的当前方法
            pbar.set_description(f"Generating [{current_method}]", refresh=False)

            # --- 读取与预处理 ---
            img_path = os.path.join(path_img, img_file)
            depth_path = os.path.join(path_depth, depth_file)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            # 注意：此处 resize 可能会大幅增加内存占用，请根据硬件调整
            img = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3))
            
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 255.0
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # --- 调用退化模型 ---
            J_out, T_out, Mask_out, A_out = simulator.degrade(
                img, depth, A=[0.7, 0.7, 0.7], mask_count=3000, method=current_method
            )
            
            cv2.imwrite(f"data/degraded/{img_file}", cv2.cvtColor((J_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            cv2.imwrite(f"data/transmission/{img_file}", (T_out * 255).astype(np.uint8))

            cv2.imwrite(f"data/mask/{img_file}", (Mask_out * 255).astype(np.uint8))

            with open(f"data/atmosphere/{img_file.split('.')[0]}.txt", 'w') as f:
                for i in range(3):
                    f.write(str(A_out[i]) + ' ')
