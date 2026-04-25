import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms as T
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import sys
import os
import time
from torchvision.transforms import ToPILImage
import argparse
import sys
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from load_data import LoadData
from Networkv4 import UnifiedRestorationNet
from utils import LossManager, PerceptualLoss, TVLoss

sys.path.append('..')
os.chdir(os.path.abspath(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_loss(data, stage='train', savepath='plots'):
    plt.figure(figsize=(8, 6))
    plt.plot(data, label=f'{stage.capitalize()}', linewidth=2.5, alpha=0.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.text(x=0, y=min(data), s=f'minimum {stage} loss: {min(data)} at epoch {data.index(min(data))}', fontsize=12)
    # plt.xscale('log')
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.legend(fontsize=14)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gcf().subplots_adjust(left=0.18, bottom=0.16)
    plt.savefig(f'{savepath}', dpi=300)
    plt.close()
    
def load_test_datas(num_data):
    '''''
    加载有随机选取的测试数据，返回一个 tuple 包含 (testI, testT, testM, testA, testLabels)
    '''
    testI = t.stack([train_dataset[i][0] for i in range(num_data//2)], dim=0)
    testI = t.cat((testI, t.stack([val_dataset[i][0] for i in range(num_data//2, num_data)], dim=0)), dim=0)
    testT = t.stack([train_dataset[i][1] for i in range(num_data//2)], dim=0)
    testT = t.cat((testT, t.stack([val_dataset[i][1] for i in range(num_data//2, num_data)], dim=0)), dim=0)
    testM = t.stack([train_dataset[i][2] for i in range(num_data//2)], dim=0)
    testM = t.cat((testM, t.stack([val_dataset[i][2] for i in range(num_data//2, num_data)], dim=0)), dim=0)
    testA = t.stack([train_dataset[i][3] for i in range(num_data//2)], dim=0)
    testA = t.cat((testA, t.stack([val_dataset[i][3] for i in range(num_data//2, num_data)], dim=0)), dim=0)
    testLabels = t.stack([train_dataset[i][4] for i in range(num_data//2)], dim=0)
    testLabels = t.cat((testLabels, t.stack([val_dataset[i][4] for i in range(num_data//2, num_data)], dim=0)), dim=0)
    
    testI = testI.to(device, dtype=t.float32)
    testT = testT.to(device, dtype=t.float32)
    testM = testM.to(device, dtype=t.float32)
    testA = testA.to(device, dtype=t.float32)
    testLabels = testLabels.to(device, dtype=t.float32)
    
    return testI, testT, testM, testA, testLabels

def write_log(savepath, message):
    ''''
    记录训练参数和训练细节到 details.txt 文件中
    '''
    with open(f"{savepath}/details.txt", 'w') as f:
        f.write(f'tesing images: {message[0]} \n')
        f.write(f'Total Train images: {message[1]}, Total Val images: {message[2]} \n')
        f.write(f'{message[3]} of data used for training \n')
        f.write(f'Epochs: {message[4]}, Batch size: {message[5]}, Save interval: {message[6]}\n')
        if message[7] is not None:
            f.write(f'loaded model from {message[7]}\n')
        f.write(f'loss weights for I, I0, T, M, A: {message[8]}\n')
        f.close()
        
def train(device, train_loader, val_loader, Testdata, test_indx, savepath, epochs=100, save_interval=25, model_path = None):
    showshell = ToPILImage() # usage showshell(img).show()
    Model = UnifiedRestorationNet().to(device)
    
    if model_path is not None:
        try:
            state_dict = t.load(model_path, weights_only=True)
            Model.load_state_dict(state_dict)
            tqdm.write('loaded model ' + model_path.split('/')[-1])
        except Exception as e:
            tqdm.write(f"Error loading model from {model_path}: {e}")
            return

    lossweights = [1, 0.5, 1, 0.05, 1] # I, I0, T, M, A 的权重
    
    criterion_mse = t.nn.MSELoss().to(device)
    criterion_l1 = t.nn.L1Loss().to(device) # 边缘保留比 MSE 好
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_tv = TVLoss().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    loss_manager = LossManager(
        Model,
        criterion_l1,
        criterion_mse,
        criterion_perceptual,
        criterion_tv,
        lossweights,
        trans_epoch=10
    )
    optimizer = t.optim.Adam(Model.parameters(), lr=5e-4, betas=[0.9, 0.999])

    savepath = savepath + f'-{Model.name}'
    os.makedirs(os.path.join(savepath, 'saved models'))
    
    lossfilename = Model.name + '_loss.txt'
    lossfilename_with_path_train = os.path.join(savepath, lossfilename)
    lossfilename_with_path_val = os.path.join(savepath, 'val_' + lossfilename)
    lossfilename_with_path_val_ssim_psnr = os.path.join(savepath, 'val_psnr_ssim' + lossfilename)

    tqdm.write(f'saving file {lossfilename_with_path_train} & {lossfilename_with_path_val}')
    
    last_loss = float("inf")  # 根据loss大小来保存
    last_loss_val = float("inf")
    best_epoch = 0
    best_epoch_val = 0
    
    messages = (test_indx, len(train_loader.dataset), len(val_loader.dataset), args.TrainToTest, epochs, args.Batch_size, save_interval, model_path, lossweights)
    write_log(savepath, messages)
    train_losses, val_losses = [], []
    psnr_list, ssim_list = [], []
    
    for epoch in range(epochs):
        Model.train()
        with tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', total=len(train_loader), mininterval=5) as pbar:
            running_loss = 0
            for batch_idx, (input, label_T, label_M, label_A, label_I) in pbar:
                input, label_T, label_M, label_A, label_I = input.to(device, dtype=t.float32), label_T.to(device, dtype=t.float32), label_M.to(device, dtype=t.float32), label_A.to(device, dtype=t.float32), label_I.to(device, dtype=t.float32)
                Model.zero_grad()
                optimizer.zero_grad()
                J, J0, T, M, A = Model(input)
                loss = loss_manager.compute(epoch, J, J0, T, M, A, input, label_I, label_T, label_M, label_A)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_loss =  running_loss / (batch_idx + 1) 
                pbar.set_postfix(loss=train_loss, refresh=False)
                
        with open(lossfilename_with_path_train, 'a') as f:
            tqdm.write(f"Time: {time.strftime('%H:%M:%S')} Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.6f}", file=f)
            train_losses.append(train_loss)
        
        Model.eval()
        with t.no_grad():   
            val_loss, val_psnr, val_ssim = 0, 0, 0
            for num, (input, T, M, A, label_I) in enumerate(val_loader):
                input, T, M, A, label_I = input.to(device, dtype=t.float32), T.to(device, dtype=t.float32), M.to(device, dtype=t.float32), A.to(device, dtype=t.float32), label_I.to(device, dtype=t.float32)
                output = Model(input)
                val_loss += loss_manager.compute(epoch, *output, input, label_I, T, M, A).item()
                val_psnr += psnr(output[0], label_I)
                val_ssim += ssim(output[0], label_I)
            val_loss /= (num + 1)
            val_psnr /= (num + 1)
            val_ssim /= (num + 1)

            with open(lossfilename_with_path_val, 'a') as f:
                tqdm.write(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.6f}", file=f)
                val_losses.append(val_loss)
            
            with open(lossfilename_with_path_val_ssim_psnr, 'a') as f:
                tqdm.write(f"Validation PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f} after Epoch {epoch + 1}", file=f)
                psnr_list.append(val_psnr)
                ssim_list.append(val_ssim)
                
            Test_out = Model(Testdata[0])
            save_show = t.cat([Testdata[0], Test_out[0], Test_out[1], Test_out[2], Test_out[3], Testdata[4]], dim=0) # input, J, J0, T, M, A, label
            Test_imgpath = os.path.join(savepath, f'result-in-epoch-{epoch +1}.jpg')
            save_image(save_show, Test_imgpath)
            
            if last_loss_val > val_loss:
                last_loss_val = val_loss
                Modelpathname = f'Best-model_val.pth'
                Modelpathname = os.path.join(savepath,'saved models', Modelpathname)
                t.save(Model.state_dict(), Modelpathname)
                tqdm.write('Saving minimum val loss in epoch %d' % (epoch + 1))
                best_epoch_val = epoch
                
            if last_loss > loss:
                last_loss = loss
                Modelpathname = f'Best-model.pth'
                Modelpathname = os.path.join(savepath,'saved models', Modelpathname)
                t.save(Model.state_dict(), Modelpathname)
                tqdm.write('Saving minimum loss in epoch %d' % (epoch + 1))
                best_epoch = epoch
            if (epoch+1) % save_interval == 0:
                t.save(Model.state_dict(), os.path.join(savepath, 'saved models', f'model-in-epoch-{epoch + 1}.pth'))
            if epoch == epochs - 1:
                t.save(Model.state_dict(), os.path.join(savepath,'saved models', 'last-model.pth'))
        t.cuda.empty_cache()
        
    with open(lossfilename_with_path_train, 'a') as f:
        tqdm.write(f"best epoch in {best_epoch + 1}", file=f)
    with open(lossfilename_with_path_val, 'a') as f:
        tqdm.write(f"best epoch in {best_epoch_val + 1}", file=f)

    plot_loss(train_losses, stage='train', savepath=f'{savepath}/train_loss.jpg')
    plot_loss(val_losses, stage='val', savepath=f'{savepath}/val_loss.jpg')
    plot_loss(psnr_list, stage='val_psnr', savepath=f'{savepath}/val_psnr.jpg')
    plot_loss(ssim_list, stage='val_ssim', savepath=f'{savepath}/val_ssim.jpg')

if __name__ == "__main__": 
    # This document is to train simulation data
    epoch = 200
    trainers = [[f'--Epoch', f'{epoch}', '--Root', 'data', '--TrainToTest', '0.8', '--Batch_size', '3', '--Save_interval', '50']]
    
    for i in range(trainers.__len__()):
        sys.argv += trainers[i]
        parser = argparse.ArgumentParser(description='Net for Speckle Experiment')
        parser.add_argument('--Root', type=str, default='data/Scatter4', help='Path to scatter data')
        parser.add_argument('--Load_model_path', type=str, default=None, help='Path to pre-trained model')
        parser.add_argument('--Epoch', type=int, default=80, help='Epoch time')
        parser.add_argument('--Batch_size', type=int, default=5, help='Batch size')
        parser.add_argument('--Save_interval', type=int, default=51, help='Save model interval')
        parser.add_argument('--TrainToTest', type=float, default=0.95, help='The ratio of Train number to Test number')
        args = parser.parse_args()

        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        savepath = f'Train_results/{current_time}'

        device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        rdseed = 4
        random.seed(rdseed)
        dataset = LoadData(root=args.Root, random_sequence=True, random_seed=rdseed , total_num=1000 ) # add total_num to limit the dataset size for quick test
        
        num_train = int(len(dataset) * args.TrainToTest)
        train_dataset = Subset(dataset, range(num_train))
        val_dataset   = Subset(dataset, range(num_train, len(dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.Batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.Batch_size, shuffle=False)
        
        print(f'total dataset size: {len(dataset)}')
        print(f"data loaded. number of train image is: {len(train_dataset)}, and val image is: {len(val_dataset)}")

        transform = T.Compose([T.ToTensor()])
        
        dataset_indx = dataset.__get_indices__()
        test_num = 8
        test_indx = dataset_indx[:test_num]
        test_indx = np.append(test_indx, dataset_indx[num_train:num_train+test_num]) # other half from test set
            
        print(f'Test image using indexes {test_indx}')
        testdata = load_test_datas(test_num)
        
        print('using device :', device)
        if device == 'cuda':
            print(f'using gpu {t.cuda.get_device_name()}')
        print( f'Root path {args.Root} ', f'Output path {savepath} ', f'Using model {args.Load_model_path} ')
        train(device, train_loader, val_loader, testdata, test_indx, savepath, args.Epoch, args.Save_interval, args.Load_model_path)
        
        print('training finished, results saved in ' + savepath)