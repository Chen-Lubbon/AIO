'''
    从data读取图像和深度，生成不同的恶劣天气图像和对应的maps，并按照9:1随机划分为train和test，保存到data文件夹
'''
    
import cv2
from utils import DegredationSimulator
import numpy as np
import os
import random
from tqdm import tqdm
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

os.chdir(os.path.dirname(__file__))  # 切换到当前脚本所在目录，确保路径正确

if __name__ == "__main__":
    
    path_img = "data/landscape"
    path_depth = "data/depth"
    
    # --- v2: 创建 train 和 test 相关的多级目录 ---
    splits = ["train", "test"]
    subdirs = ['landscape', "degraded", "transmission", "mask", "atmosphere"]
    
    for split in splits:
        for subdir in subdirs:
            os.makedirs(f"data/{split}/{subdir}", exist_ok=True)
    
    images = os.listdir(path_img)
    # 修复了原代码 '. png' 的小笔误
    images = [img for img in images if img.endswith('.jpg') or img.endswith('.png')]
    images.sort(key=natural_sort_key)
    
    depths = os.listdir(path_depth)
    depths = [d for d in depths if d.endswith('.jpg') or d.endswith('.png')]
    depths.sort(key=natural_sort_key)  # 确保图像和深度图的顺序一致
    
    print(f"Found {len(images)} images in {path_img}.")
    print(f"Found {len(depths)} depth maps in {path_depth}.")
    
    # --- v2: 随机抽取 10% 的索引作为 test 集 ---
    total_images = len(images)
    test_ratio = 0.1
    num_test = int(total_images * test_ratio)
    # 固定随机种子（可选），如果你希望每次运行抽取的图片都一样，取消注释下一行
    random.seed(42) 
    test_indices = set(random.sample(range(total_images), num_test))
    print(f"Split {total_images - num_test} images for train, {num_test} images for test.")
    
    change_interv = total_images // 6
    print(f"Changing degradation type every {change_interv} images.")
    
    simulator = DegredationSimulator()
    methods = ["fog", "rain", "snow", "underwater1", "underwater2", "underwater3"]
    print(f"Starting degradation with method: {methods[0]}")
    
    with tqdm(enumerate(zip(images, depths)), desc='Generating', total=total_images, mininterval=5) as pbar:
        for index, (img_file, depth_file) in pbar:
            # 计算当前应该使用的 method 索引
            method_index = min(index // change_interv, len(methods) - 1)
            current_method = methods[method_index]
            
            # --- v2: 确定当前图片应该放入 train 还是 test 文件夹 ---
            split_dir = "test" if index in test_indices else "train"
            
            # 更新进度条显示的当前方法和归属
            pbar.set_description(f"Generating [{current_method}] -> {split_dir}", refresh=False)

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
            
            # --- v2: 保存到对应 split 的目录，并保持原文件名 ---
            base_save_path = f"data/{split_dir}"
            
            cv2.imwrite(f"{base_save_path}/degraded/{img_file}", cv2.cvtColor((J_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{base_save_path}/transmission/{img_file}", (T_out * 255).astype(np.uint8))
            cv2.imwrite(f"{base_save_path}/mask/{img_file}", (Mask_out * 255).astype(np.uint8))
            cv2.imwrite(f"{base_save_path}/landscape/{img_file}", cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            txt_filename = f"{img_file.split('.')[0]}.txt"
            with open(f"{base_save_path}/atmosphere/{txt_filename}", 'w') as f:
                for i in range(3):
                    f.write(str(A_out[i]) + ' ')