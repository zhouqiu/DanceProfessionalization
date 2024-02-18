import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import math
import numpy as np
from .blocks import ConvBlock, LinearBlock
from utils.util import mkdir
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MusicMotionGen(nn.Module):
    def __init__(self, kernel_size, in_c_v, nhead=4, nlayers=2):
        super(MusicMotionGen, self).__init__()

        stride = 2
        pad_type = 'reflect'
        acti = 'relu'
        norm = 'none'

        layer = ConvBlock(kernel_size, in_c_v, 64,
                          stride=stride, pad_type=pad_type, norm=norm, acti=acti)
        layer += ConvBlock(kernel_size, 64, 128,
                           stride=stride, pad_type=pad_type, norm=norm, acti=acti)
        layer += ConvBlock(kernel_size, 128, 256,
                           stride=stride, pad_type=pad_type, norm=norm, acti=acti)
        self.enc_conv = nn.Sequential(*layer)


        encoder_layer = TransformerEncoderLayer(256, nhead, 256)
        self.trans_encoder = TransformerEncoder(encoder_layer, nlayers)

        self.upsample = nn.Upsample(scale_factor=2, mode='linear',align_corners=False)

        layer = LinearBlock(256, 128, norm=norm, acti=acti)
        self.dec_conv1 = nn.Sequential(*layer)
        layer = LinearBlock(128, 64, norm=norm, acti=acti)
        self.dec_conv2 = nn.Sequential(*layer)
        layer = LinearBlock(64, in_c_v, norm=norm, acti='none')
        self.dec_conv3 = nn.Sequential(*layer)

        self.avg_pool = nn.AvgPool2d(2, stride=2)
        # self.max_pool = nn.MaxPool1d(2, stride=2)


    def min_pool(self, mask, ks=2):#[B,T]
        b = mask[:, 1::ks]
        c = mask[:, ::ks]
        d = torch.min(b,c) #[B,T/2]
        return d


    def forward(self, content, target_matrix, attn_mask=None, src_key_padding_mask=None):
        if attn_mask is not None:
            attn_mask1 = self.avg_pool(attn_mask.unsqueeze(0).unsqueeze(0)) * 2
            attn_mask2 = self.avg_pool(attn_mask1) * 2
            attn_mask3 = self.avg_pool(attn_mask2) * 2
            attn_mask3 = attn_mask3.squeeze(0).squeeze(0) #[T/8, T/8]
        else:
            attn_mask3 = None

        if src_key_padding_mask is not None:
            src_key_padding_mask1 = src_key_padding_mask[..., ::2] #[B,T/2]
            src_key_padding_mask2 = src_key_padding_mask1[..., ::2]  # [B,T/4]
            src_key_padding_mask3 = src_key_padding_mask2[..., ::2]  # [B,T/8]
            src_key_padding_mask3 = src_key_padding_mask3.bool() #[B,T/8]
        else:
            src_key_padding_mask3 = None

        content = self.enc_conv(content)  # [B, 256, T/8]
        content = self.trans_encoder(content.permute(2,0,1),
                                     mask=attn_mask3,
                                     src_key_padding_mask=src_key_padding_mask3).permute(1,2,0)   # [B, 256, T/8]

        matrix1 = self.avg_pool(target_matrix.unsqueeze(1)) * 2 #[B,1,T/2,T/2]
        matrix2 = self.avg_pool(matrix1) * 2 #[B,1,T/4,T/4]
        matrix3 = self.avg_pool(matrix2) * 2  # [B,1,T/8,T/8]

        latent_v = torch.bmm(content, matrix3.squeeze(1))  # [B, 256, T/8]

        out = self.upsample(latent_v)# [B, 256, T/4]
        out = self.dec_conv1(out.permute(0,2,1)).permute(0,2,1) # [B, 128, T/4]
        out = self.upsample(out) # [B, 128, T/2]
        out = self.dec_conv2(out.permute(0, 2, 1)).permute(0, 2, 1) # [B, 64, T/2]
        out = self.upsample(out)  # [B, 64, T]
        out = self.dec_conv3(out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, in_c_v, T]


        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool()
            out = out.masked_fill(
                src_key_padding_mask.unsqueeze(1),
                float(0),
            )

        return out, latent_v


class AISTModel(nn.Module):
    def __init__(self, opt):
        super(AISTModel, self).__init__()
        self.opt = opt
        self.phase = opt.phase
        self.kernel_size = opt.kernel_size
        print("kernel_size: {}".format(self.kernel_size))
        self.isFinetune = opt.isFinetune
        print("isFinetune: {}".format(self.isFinetune))

        self.model_dir = os.path.join(opt.net_root, opt.net_path)  # './all_nets/net'
        mkdir(self.model_dir)

        gpu_ids = opt.gpu_ids
        self.device_count = len(gpu_ids)
        print("device_cont:{}".format(self.device_count))

        self.gen = MusicMotionGen(self.kernel_size, 60)
        self.gen.apply(self.__weights_init('kaiming'))
        self.gen = nn.DataParallel(self.gen)
        self.gen.cuda()

        gen_params = list(self.gen.parameters())
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=opt.lr_gen, betas=(0.9, 0.999))

        self.rec_w = opt.rec_w
        self.velo_w = opt.velo_w
        print("velo_w:{}".format(self.velo_w))


        self.total_length = opt.total_length

        self.loss_dict = {}

        if self.phase != 'train' or opt.continueTrain:
            if opt.continueTrain:
                print("continue train: from epoch {}.".format(opt.model_epoch))
            epoch = opt.model_epoch
            print("Load network: {}".format(epoch))

            gen_name = os.path.join(self.model_dir, 'gen_%08d.pt' % epoch)
            state_dict = torch.load(gen_name)
            self.gen.module.load_state_dict(state_dict)

            opt_name = os.path.join(self.model_dir, 'g_optimizer.pt')
            state_dict = torch.load(opt_name)
            self.gen_opt.load_state_dict(state_dict)

            print('load finish.')


    def __update_dict(self, old_dict, new_dict):
        for key, value in new_dict.items():
            old_dict[key] = value

    def __weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find(
                    'Linear') == 0) and hasattr(m, 'weight'):
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun

    @staticmethod
    def recon_criterion(predict, target, padding_mask=None):
        if padding_mask is not None:
            predict = predict.masked_fill(
                padding_mask.bool().unsqueeze(1),
                float(0),
            )
            target = target.masked_fill(
                padding_mask.bool().unsqueeze(1),
                float(0),
            )
        return torch.mean(torch.abs(predict - target))


    def make_attn_mask(self, times=1, pad=50):
        negetive_infinity = -100000.0
        attn_mask = negetive_infinity * torch.ones((self.total_length, self.total_length))
        for i in range(self.total_length):
            start = max(0, i - pad)
            finish = min(self.total_length, i + pad)
            attn_mask[i, start: finish] = torch.zeros(finish - start)
        return attn_mask.repeat(times,1)

    def get_glb(self, direct, off_len, parents, padding_mask=None): #[B, 60, T]  [B,20]  [B,20] [B,T]
        B,C,T = direct.size()
        J = C//3
        direct = direct.permute(0,2,1).reshape((B, T, J, -1)) #[B,T,J,3]
        direct_len = torch.norm(direct, dim=-1, keepdim=True)#[B,T,J,1]
        direct_len = torch.where(direct_len == 0, 1e-09 * torch.ones_like(direct_len), direct_len)
        direct = direct / direct_len  # normalization

        direct = direct * off_len.unsqueeze(1).unsqueeze(-1) # [B, T, 20, 3]
        glb = torch.zeros((B, T, J + 1, 3)).cuda()  # [B, T, 21, 3]

        for j in range(J):
            c = j + 1
            p = parents[0, j].long() #[1]
            parent_pos = torch.index_select(glb, 2, p).squeeze(2)

            glb[..., c, :] = parent_pos + direct[..., j, :]

        glb = glb[..., 1:, :]  # [B, T, 20, 3]
        glb = glb.reshape((B,T,-1)).permute(0,2,1) #[B, 60, T]

        if padding_mask is not None:
            glb = glb.masked_fill(
                padding_mask.bool().unsqueeze(1),
                float(0),
            )
        return glb  #[B, 60, T]

    def cal_new_correlation(self,corr,real_length): #{B,Tm,Ta]
        B,T,_ = corr.size()
        total_corr = torch.zeros_like(corr).cuda() + torch.eye(T).expand((B,T,T)).cuda()
        for b in range(B):
            real_len = int(real_length[b])
            real_corr = corr[b, :real_len, :real_len]
            real_corr = real_corr / torch.sum(real_corr, dim=-2, keepdim=True)
            total_corr[b, :real_len, :real_len] = real_corr
        return total_corr

    def set_input(self, data):

        self.matrix = data["matrix"].clone().detach().float().cuda()  # [B, Tm, Ta]

        self.direct_in = data["ndirect"].clone().detach().float().cuda()  # [B, 60, T]
        self.rt_in = data["nrtpos"].clone().detach().float().cuda()  # [B, 3, T]
        self.direct_pro = data["pdirect"].clone().detach().float().cuda()  # [B, 60, T]
        self.rt_pro = data["prtpos"].clone().detach().float().cuda()  # [B, 3, T]

        self.attn_mask = self.make_attn_mask(self.device_count).cuda()  # [T, T]
        self.padding_mask = data["padding_mask"].clone().detach().float().cuda()  # [B, T]
        self.real_length = data["real_length"].clone().detach().float().cuda()  # [1]
        self.corr = data["dtw_matrix"].clone().detach().float().cuda()  # [B, Tm, Ta]
        self.feat = data["dtw_dist"].clone().detach().float().cuda()  # [B, Tm, Ta]

        self.off_len = data["off_len"].clone().detach().float().cuda() # [B,20]
        self.parents = data["parents"].clone().detach().float().cuda() # [B,20]

        self.ft = data["ft"].clone().detach().float().cuda() #[B,4,T]
        self.correlation = self.cal_new_correlation(self.corr,self.real_length) # [B, Tm, Ta]



    def forward(self, data):
        self.set_input(data)
        if not self.isFinetune:
            self.direct_out, self.latent_code = self.gen(self.direct_in,
                                                         self.matrix,
                                                         self.attn_mask,
                                                         self.padding_mask)
        else:
            self.direct_out, self.latent_code = self.gen(self.direct_in,
                                                         self.correlation,
                                                         self.attn_mask,
                                                         self.padding_mask)

    def backward_G(self):

        l_rec = self.recon_criterion(self.direct_out, self.direct_pro, self.padding_mask)
        self.glb_out = self.get_glb(self.direct_out, self.off_len, self.parents, padding_mask=self.padding_mask)
        self.glb_pro = self.get_glb(self.direct_pro, self.off_len, self.parents, padding_mask=self.padding_mask)
        l_velo = self.recon_criterion(self.glb_out[..., :-1]-self.glb_out[..., 1:],
                                      self.glb_pro[..., :-1]-self.glb_pro[..., 1:],
                                      self.padding_mask[..., 1:])

        l_total = self.rec_w * l_rec + self.velo_w * l_velo
        l_total.backward()

        ret_dict = {
            'rec': l_rec,
            'velo': l_velo
        }
        return ret_dict

    def optimize(self):
        # G
        self.gen_opt.zero_grad()
        gen_loss_dict = self.backward_G()
        self.__update_dict(self.loss_dict, gen_loss_dict)
        self.gen_opt.step()

    def test(self, data):
        '''For producing results'''
        self.eval()
        self.forward(data)
        self.cuda()

        if not self.isFinetune:
            ortpos = torch.bmm(self.rt_in, self.matrix)
            oft = torch.bmm(self.ft, self.matrix)
            oft = torch.round(oft)
        else:
            ortpos = torch.bmm(self.rt_in, self.correlation)
            oft = torch.bmm(self.ft, self.correlation)
            oft = torch.round(oft)


        out_dict = {
            "target_mat": self.matrix,  # [B, T, T]
            "pred_mat": self.corr,  # [B, T, T]
            "pred_feat": self.feat,

            "direct_in": self.direct_in,  # [B, 60, T]
            "direct_out": self.direct_out,  # [B, 60, T]
            "direct_pro": self.direct_pro,  # [B, 60, T]
            "nrtpos": self.rt_in,  # [B, 3 , T]
            "ortpos": ortpos,
            "prtpos": self.rt_pro,  # [B, 3 , T]

            "real_length": self.real_length,  # [1]

            "oft": oft,
            "nft": self.ft
        }

        return out_dict


    def getlossDict(self):
        return self.loss_dict

    def save_networks(self, epoch):
        gen_name = os.path.join(self.model_dir, 'gen_%08d.pt' % epoch)
        torch.save(self.gen.module.state_dict(), gen_name)

        opt_name = os.path.join(self.model_dir, 'g_optimizer.pt')
        torch.save(self.gen_opt.state_dict(), opt_name)








