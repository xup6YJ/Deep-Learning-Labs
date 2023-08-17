import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
import json
from datetime import datetime

from torchvision import datasets, models, utils

import torch.backends.cudnn as cudnnbackend
from torch.utils.tensorboard import SummaryWriter



def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.current_epoch = current_epoch
        self.type = args.kl_anneal_type

        if self.type != 'None':
            self.t_beta = self.frange_cycle_linear(n_epoch=args.num_epoch, start_epoch=args.kl_anneal_start_epoch, 
                                                   n_cycle=args.kl_anneal_cycle, ratio = args.kl_anneal_ratio)
        else:
            self.t_beta = np.ones(args.num_epoch)


        
    def update(self):
        # TODO
        self.current_epoch += 1

    
    def get_beta(self):
        # TODO
        beta = self.t_beta[self.current_epoch]

        return beta

    # def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=1):
    #     # TODO
    #     raise NotImplementedError

    # def frange_cycle_linear(self, n_epoch, start_epoch = 10, start = 0.1, stop = 1,  n_cycle=4, ratio=0.5):
    #     # TODO
    #     if self.type == 'Cyclical':
    #         n_cycle = n_cycle
    #     elif self.type == 'Monotonic':
    #         n_cycle = 1

    #     L = np.ones(n_epoch)
    #     period = n_epoch/n_cycle
    #     step = (stop-start)/(period*ratio) # linear schedule

    #     for c in range(n_cycle):
    #         v , i = start , 0

    #         while v <= stop and (int(i+c*period) < n_epoch):
    #             L[int(i+c*period)] = v
    #             v += step
    #             i += 1
    #     return L     

    def frange_cycle_linear(self, n_epoch = 100, start_epoch = 10, start = 0, stop = 1,  n_cycle=4, ratio=0.5):
        # TODO
        if self.type == 'Cyclical':
            n_cycle = n_cycle
        elif self.type == 'Monotonic':
            n_cycle = 1

        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                if i < start_epoch:
                    L[i] = start
                else:
                    L[int(i+c*period)] = v
                    v += step
                i += 1
        return L  

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):

        min_loss = np.inf
        c_path = os.getcwd()
        r_path = os.path.join(c_path, 'record')
        if not os.path.exists(r_path):
            os.makedirs(r_path)
        e_path = os.path.join(r_path, self.args.name)
        if not os.path.exists(e_path):
            os.makedirs(e_path)

        self.logger = SummaryWriter(e_path)

        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            # adapt_TeacherForcing = True 
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, psnr = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), psnr, lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), psnr, lr=self.scheduler.get_last_lr()[0])
            
            
            self.logger.add_scalar(f'train/loss', loss, self.current_epoch)
            self.logger.add_scalar(f'train/psnr', psnr, self.current_epoch)
            # self.logger.add_scalar(f'train/MSE', train_mse, self.current_epoch)
            # self.logger.add_scalar(f'train/KL', kld, self.current_epoch)
            self.logger.add_scalar(f'train/beta', self.beta, self.current_epoch)
            # if loss < min_loss:
            #     min_loss = loss
            #     print('current training loss < min_loss')
            #     self.save(os.path.join(self.args.save_root, f"{args.time}_epoch_{self.current_epoch}_loss_{loss:.5f}.ckpt"))
                
            self.eval()

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"PSNR_{self.val_psnr:.3f}_epoch_{self.current_epoch}.ckpt"))


            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, val_psnr = self.val_one_step(img, label)
            self.val_psnr = val_psnr

            self.tqdm_bar('val', pbar, loss.detach().cpu(), val_psnr, lr=self.scheduler.get_last_lr()[0])
            self.logger.add_scalar(f'val/loss', loss, self.current_epoch)
            self.logger.add_scalar(f'val/psnr', val_psnr, self.current_epoch)
    
    def training_one_step(self, img, label, adapt_TeacherForcing = True):
        # TODO
        # return loss

        self.frame_transformation.train().zero_grad()
        self.label_transformation.train().zero_grad()
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor.train().zero_grad()
        self.Decoder_Fusion.train().zero_grad()
        # Generative model
        self.Generator.train().zero_grad()

        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        
        
        train_mse = 0
        kld = 0
        PSNR_LIST = []
        for i in range(1, self.train_vi_len):

            x_t_1 = img[i-1]
            x_t = img[i]

            #pose encoder
            label_feat = self.label_transformation(label[i])
            #frame encoder
            if i == 1:
                human_feat_hat = self.frame_transformation(x_t_1)
            else:
                if adapt_TeacherForcing:
                    #put label into training
                    human_feat_hat = self.frame_transformation(x_t_1)
                else:
                    #put prediction into training
                    human_feat_hat = self.frame_transformation(self.pre_prediction)
            
            #put output 
            #frame encoder
            frame_out = self.frame_transformation(x_t)
            z, mu_p, logvar_p = self.Gaussian_Predictor(frame_out, label_feat)  
            
            #Put output of pose ecoder, frame encoder, z in decoder fusion 
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)  
            #put output of decoder into generator
            x_hat_t = self.Generator(parm)
            self.pre_prediction = x_hat_t  

            #loss
            #MSE
            mse_sub = self.mse_criterion(x_t, x_hat_t)
            train_mse += mse_sub

            #KL D
            kld += self.kl_criterion(mu_p, logvar_p, self.args.batch_size)

            #PSNR
            PSNR = self.Generate_PSNR(x_t, x_hat_t)
            PSNR_LIST.append(PSNR.item())


        self.beta = self.kl_annealing.get_beta()
        loss = train_mse + kld * self.beta
        avg_loss = loss / self.train_vi_len
        avg_PSNR = sum(PSNR_LIST)/(len(PSNR_LIST)-1)

        loss.backward()
        self.optim.step()  

        return avg_loss, avg_PSNR

    
    # def val_one_step(self, img, label):
    #     # TODO
    #     raise NotImplementedError

    def val_one_step(self, img, label):
        # return loss
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        # assert label.shape[0] == 630, "Testing pose seqence should be 630"
        # assert img.shape[0] == 1, "Testing video seqence should be 1"
        
        # decoded_frame_list = [img[0].cpu()]
        # label_list = []

        # Normal normal
        last_human_feat = self.frame_transformation(img[0])
        first_templete = last_human_feat.clone()
        out = img[0]
        
        self.best_val_psnr = np.NINF
        val_mse = 0
        val_PSNR_LIST = []
        for i in range(1, self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            label_feat = self.label_transformation(label[i])

            x_t_1 = img[i-1]
            x_t = img[i]

            if i == 1:
                human_feat_hat = self.frame_transformation(x_t_1)
            else:
                human_feat_hat = self.frame_transformation(self.val_previous)

            
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            x_hat_t = self.Generator(parm)
            self.val_previous = x_hat_t

            #loss
            #MSE
            mse_sub = self.mse_criterion(x_t, x_hat_t)
            val_mse += mse_sub

            #PSNR
            PSNR = self.Generate_PSNR(x_t, x_hat_t)
            val_PSNR_LIST.append(PSNR.item())

        loss = val_mse
        avg_loss = loss / self.val_vi_len
        avg_psnr = sum(val_PSNR_LIST)/(len(val_PSNR_LIST)-1)

        if avg_psnr > self.best_val_psnr:
            self.plot_psnr(val_PSNR_LIST)

        return avg_loss, avg_psnr
    
    def plot_psnr(self, result):
        plt.figure(figsize=(10, 5))
        frames = range(1, len(result) +1)
        avg = sum(result)/(len(result)-1)

        plt.plot(frames, result, 'r', label = f'AVG_PSNR:{avg:.3f}')
        
        plt.title(f'Per frame Quality (PSNR)')
        plt.legend()
        plt.xlabel('Frames')
        plt.ylabel('PSNR')
        # plt.figure()
        plt.savefig(os.path.join(self.args.save_root, 'Frame_PSNR.jpeg'))
        # plt.show()
        plt.close()

    def teacher_forcing_ratio_update(self):
        # TODO

        '''
        Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        '--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
        '--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
        '--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
        '--ckpt_path',     type=str,    default=None,help="The path of your checkpoints") 
        '''
        
        
        if self.current_epoch >= args.tfr_sde:
            # slope = (1.0 - args.tfr_lower_bound) / (args.niter - args.tfr_sde)
            # tfr = 1.0 - (self.current_epoch - args.tfr_sde) * slope

            tfr = 1.0 - (self.current_epoch / self.args.num_epoch) * args.tfr_d_step 
            tfr = max(tfr, 0)
            self.tfr = tfr
            print(f'Teacher ratio: {self.tfr}')

            self.logger.add_scalar(f'train/tfr', self.tfr, self.current_epoch)

        # raise NotImplementedError
    '''
    add
    '''
    def kl_criterion(self, mu, logvar, batch_size):

        kld = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
        return kld / batch_size

    def generate_Ground_truth(self, idx):
        loaded_2D_mat = np.loadtxt(f'Demo_Test/GT{idx}.csv')
        img = torch.from_numpy(loaded_2D_mat).reshape(1, 630 ,3, 32, 64)
        return img

    def make_ground_truth_gif(self, save_root, img_tensor, idx):
        new_list = []
        for i in range(630):
            new_list.append(transforms.ToPILImage()(img_tensor[i]))
            
        new_list[0].save(os.path.join(save_root, f'seq{idx}/ground_truth.gif'), \
            format="GIF", append_images=new_list,
                    save_all=True, duration=20, loop=0)

    def Generate_PSNR(self, imgs1, imgs2, data_range=1.):
        """PSNR for torch tensor"""
        mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
        psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
        return psnr


    def calculate_PSNR(self, save_root, gen_image, idx):
        ground_truth = self.generate_Ground_truth(idx)[0]
        self.make_ground_truth_gif(save_root, ground_truth, idx)
        gen_image = gen_image[0]
        
        PSNR_LIST = []
        for i in range(1, 630):
            PSNR = Generate_PSNR(ground_truth[i], gen_image[i])
            PSNR_LIST.append(PSNR.item())
            
        return sum(PSNR_LIST)/(len(PSNR_LIST)-1)
    
    ################################################################
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
            
    def tqdm_bar(self, mode, pbar, loss, psnr, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.6f}, loss {float(loss)}" , refresh=False)
        # pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.set_postfix(PSNR=float(psnr), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=8)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str,  default = '/media/bspubuntu/3TBNAS/DLP/Lab4/LAB4/LAB4_dataset', help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=2,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true', default=True)
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=4,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    parser.add_argument('--kl_anneal_start_epoch',    type=int, default=10,              help="")
    

    args = parser.parse_args()

    now = datetime.now() 
    dt_string = now.strftime("%Y%m%d_%H%M")
    dt_string = now.strftime("%Y%m%d_%H%M")
    args.time = dt_string
    args.name = args.time + '_' + args.kl_anneal_type + '_tfr_d_step' + str(args.tfr_d_step) + 'KLratio_' + str(args.kl_anneal_ratio)+  '_weight'
    path = os.getcwd()
    args.save_root = os.path.join(path, args.name)
    # os.makedirs(args.save_root, exist_ok=True)

    print(json.dumps(vars(args), indent=2))

    main(args)
