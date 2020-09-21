import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_ssim import *
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss_new import GeneratorLoss
from model_new import *

import torch.nn as nn
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--dataset_name', default='DIV2K', type=str)

name = 'combine'
path = 'drive/My Drive/Aerocosmos/10june/esrgan/' + name + '/'

if not os.path.exists(path):
            os.makedirs(path)



l2_loss = nn.MSELoss().cuda()


if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    DATASET_NAME = opt.dataset_name
    
    train_set = TrainDatasetFromFolder('drive/My Drive/Aerocosmos/data/' + DATASET_NAME + '/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('drive/My Drive/Aerocosmos/data/'  + DATASET_NAME + '/valid', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0005)
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0005)

    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()

            


    #netG.load_state_dict(torch.load('drive/My Drive/Aerocosmos/10june/esrgan/combine/epochs/netG_epoch_4_35.pth'))
    #netD.load_state_dict(torch.load('drive/My Drive/Aerocosmos/10june/esrgan/combine/epochs/netD_epoch_4_35.pth'))
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    print(1)
    for epoch in range(1, NUM_EPOCHS + 1):
        
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        
        for data, target in train_bar:

            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((batch_size, 1, 1, 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((batch_size, 1, 1, 1))), requires_grad=False)
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
    
            netD.zero_grad()
            real_out = netD(real_img)[0].mean()
            fake_out = netD(fake_img)[0].mean()

            pred_real = netD(real_img)[1]
            pred_fake = netD(fake_img.detach())[1]

            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            #d_loss = 1 - real_out + fake_out
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()

            pred_real = netD(real_img)[1].detach()
            pred_fake = netD(fake_img)[1]

            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            
           
            g_loss = generator_criterion(fake_out, fake_img, real_img, loss_GAN)

            #g_loss = text_loss
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img)[0].mean()

            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = path+ 'training_results/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            i = 0
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
                    
                i+=1
                if i == 15:
                  break
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                if index == 1:
                  image = utils.make_grid(image, nrow=3, padding=5)
                  utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                  index += 1
    
        # save model parameters
        out_path_model = path+  'epochs/'
        
        # check path exist
        if not os.path.exists(out_path_model):
            os.makedirs(out_path_model)

        if epoch % 5 == 0:
            torch.save(netG.state_dict(), out_path_model + 'netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), out_path_model + 'netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        
        if epoch >= 0:
            out_path_stat = path+ 'statistics/'
            # check path exist 
            if not os.path.exists(out_path_stat):
                os.makedirs(out_path_stat)
            data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                          'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(1, epoch + 1))
            data_frame.to_csv(out_path_stat + name + '_train_results.csv', index_label='Epoch')
