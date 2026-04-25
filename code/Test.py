import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms as T
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse
from torch.utils.data import Subset

from Networkv2 import UnifiedRestorationNet
from load_data import LoadData

# sys.path.append('..')
os.chdir(os.path.abspath(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test(device, test_loader, savepath, model_path, train_dataset_name):
    os.makedirs(savepath, exist_ok=True)
    
    Model = UnifiedRestorationNet().to(device)
    Model.load_state_dict(torch.load(model_path))
    Model.eval()
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = torch.nn.functional.mse_loss
    ssim_file = f'{savepath}/ssim for {train_dataset_name} dataset.txt'
    psnr_file = f'{savepath}/psnr for {train_dataset_name} dataset.txt'
    psnr_total = 0.0
    ssim_total = 0.0
    
    with torch.no_grad() and open(ssim_file, 'w') as ssim_f, open(psnr_file, 'w') as psnr_f:
        progress_bar = tqdm(
        test_loader,
        desc="Testing",
        mininterval=1.0, maxinterval=5.0, leave=False
    )
        num_digits = len(str(len(test_loader)))
        
        for num, (input, T, M, A, label_I) in enumerate(test_loader):
            
            input = input.to(device, dtype=torch.float32)
            T = T.to(device, dtype=torch.float32)
            M = M.to(device, dtype=torch.float32)
            A = A.to(device, dtype=torch.float32)
            label_I = label_I.to(device, dtype=torch.float32)

            output = Model(input)

            ssim_value = ssim(output, label_I).item()
            psnr_value = psnr(output, label_I).item()
            ssim_f.write(f'{ssim_value}\n')
            psnr_f.write(f'{psnr_value}\n')
            psnr_total += psnr_value
            ssim_total += ssim_value
            # 保存结果图像
            padded_idx = str(num).zfill(num_digits)
            save_image(output, os.path.join(savepath, f'{int(padded_idx)}.png'))
            # save_image(input, os.path.join(savepath, f'{padded_idx}_input.png'))
            # save_image(label_I, os.path.join(savepath, f'{padded_idx}_label.png'))

            progress_bar.set_postfix(
                ssim=ssim_value, 
                psnr=psnr_value, 
                avg_ssim=ssim_total/(num+1), 
                avg_psnr=psnr_total/(num+1),
                refresh=False
            )
    ssim_f.close()
    psnr_f.close()

if __name__ == "__main__":
    testers = [['--root_data', 'data', '--model_path', 'checkpoints/LOTDataset/best_model.pth', '--savepath', 'Test_results', '--TrainToTest', '0.9']]
    
    sys.argv += testers[0]
    parser = argparse.ArgumentParser(description="Test U-shape Transformer for Underwater Image Enhancement")
    parser.add_argument('--root_data', type=str, required=True, help='Path to the data root directory')
    parser.add_argument('--random_sequence', type=bool, default=False, help='Randomize the sequence of the dataset')
    parser.add_argument('--total_num', type=int, default=100, help='Total number of images in the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--savepath', type=str, default='Test results/Local', help='Path to save the test results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--TrainToTest', type=float, default=0.9, help='Batch size for testing')
    args = parser.parse_args()
    savepath = f'{args.savepath}'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = LoadData(args.root_data, random_sequence=args.random_sequence, total_num=args.total_num)
    num_train = int(len(dataset) * args.TrainToTest)
    test_loader = Subset(dataset, range(num_train, len(dataset)))
    test_loader = DataLoader(test_loader, batch_size=args.batch_size)
    print("data loaded")
    print(f'savepath: {savepath}')
    test(device, test_loader, savepath, args.model_path)
