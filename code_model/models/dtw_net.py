import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import math
import numpy as np
from models.blocks import ConvBlock, LinearBlock
from torch.nn.modules.loss import _Loss
from utils.util import mkdir
from torch.nn import TransformerEncoder, TransformerEncoderLayer




class DTWEncTrans(nn.Module):
    def __init__(self, in_ks, in_c_q, in_c_k, hid_c,isnorm=False,isConv=True, isTrans=False):
        super(DTWEncTrans, self).__init__()

        self.isConv = isConv
        self.isTrans = isTrans
        stride = 1
        pad_type = 'reflect'
        acti = 'lrelu'
        if isnorm:
            norm='in'
        else:
            norm = 'none'

        if self.isConv:
            layer = ConvBlock(in_ks, in_c_q, in_c_q,
                              stride=stride, pad_type=pad_type, norm=norm, acti=acti)
            layer += ConvBlock(in_ks, in_c_q, hid_c,
                               stride=stride, pad_type=pad_type, norm=norm, acti=acti)
            layer += ConvBlock(in_ks, hid_c, hid_c,
                               stride=stride, pad_type=pad_type, norm=norm, acti=acti)
            self.q_enc_conv = nn.Sequential(*layer)

            layer = ConvBlock(in_ks, in_c_k, in_c_k,
                              stride=stride, pad_type=pad_type, norm=norm, acti=acti)
            layer += ConvBlock(in_ks, in_c_k, hid_c,
                               stride=stride, pad_type=pad_type, norm=norm, acti=acti)
            layer += ConvBlock(in_ks, hid_c, hid_c,
                               stride=stride, pad_type=pad_type, norm=norm, acti=acti)
            self.k_enc_conv = nn.Sequential(*layer)

        if self.isTrans:
            nhead = 4
            dropout = 0.5
            nlayers = 2
            if not self.isConv:
                layer = LinearBlock(in_c_q, hid_c, norm='none', acti=acti)
                self.q_linear = nn.Sequential(*layer)
                layer = LinearBlock(in_c_k, hid_c, norm='none', acti=acti)
                self.k_linear = nn.Sequential(*layer)

            q_encoder_layers = TransformerEncoderLayer(hid_c, nhead, hid_c, dropout)
            k_encoder_layers = TransformerEncoderLayer(hid_c, nhead, hid_c, dropout)
            self.q_transformer_encoder = TransformerEncoder(q_encoder_layers, nlayers)
            self.k_transformer_encoder = TransformerEncoder(k_encoder_layers, nlayers)


    def Euclidean_dist(self, music_feat, motion_feat):#[B,T,c] [B,T, c]
        b = motion_feat.size()[0]
        c = motion_feat.size()[-1]
        M, N = music_feat.size()[1], motion_feat.size()[1]
        a = music_feat.unsqueeze(1).expand((b, M, N, c)) # [B, 1, T,c] -> [B, T,T,c]   pro
        b = motion_feat.unsqueeze(2).expand((b, M, N, c)) # [B, T, 1,c] -> [B, T,T,c]  non

        dist = torch.sqrt(torch.sum(torch.pow(a-b, 2),dim=-1)) #[B,T,T]
        dist = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))  # normalization

        return dist #[B, Tm, Ta]


    def dtw_path(self, dist, real_length):  # [B, T, T]  [B]
        distnp = dist.detach().cpu().numpy() # [B, T, T]
        (batch, M,N) = distnp.shape

        cost = np.zeros_like(distnp)   # [B, T, T]
        path = -1 * np.ones((batch, M, N, 2))

        cost[:, 0, 0] = distnp[:, 0, 0]

        for b in range(batch):
            for i in range(1, int(real_length[b])):
                cost[b, i, 0] = cost[b, i - 1, 0] + distnp[b, i, 0]
                path[b, i, 0, 0] = i - 1
                path[b, i, 0, 1] = 0

        for b in range(batch):
            for j in range(1, int(real_length[b])):
                cost[b, 0, j] = cost[b, 0, j - 1] + distnp[b, 0, j]
                path[b, 0, j, 0] = 0
                path[b, 0, j, 1] = j - 1

        # Populate rest of cost matrix within window
        for b in range(batch):
            for i in range(1, int(real_length[b])):
                for j in range(1, int(real_length[b])):
                    choices = cost[b, i - 1, j - 1], cost[b, i, j - 1], cost[b, i - 1, j]
                    cost[b, i, j] = min(choices) + distnp[b, i, j]
                    if min(choices) == cost[b, i - 1, j - 1]:
                        path[b, i, j, 0] = i - 1
                        path[b, i, j, 1] = j - 1
                    elif min(choices) == cost[b, i, j - 1]:
                        path[b, i, j, 0] = i
                        path[b, i, j, 1] = j - 1
                    else:
                        path[b, i, j, 0] = i - 1
                        path[b, i, j, 1] = j

        path01 = np.zeros_like(distnp)   # [B, T, T]
        for b in range(batch):
            path01[b, int(real_length[b])-1, int(real_length[b])-1] = 1
            ix = int(path[b, int(real_length[b])-1, int(real_length[b])-1, 0])
            iy = int(path[b, int(real_length[b])-1, int(real_length[b])-1, 1])
            while (ix >= 0 and iy >= 0):
                path01[b, ix, iy] = 1
                x = int(path[b, ix, iy, 0])
                y = int(path[b, ix, iy, 1])
                ix = x
                iy = y

        return path01 #[B, Tmr, Tar], cost[-1, -1], cost_target


    def forward(self, vec, music, real_length, attn_mask=None, src_key_padding_mask=None, cal_path=False):
        # enc
        q = vec.permute(0,2,1) #[B,T,in_c_q]
        k = music.permute(0,2,1) #[B,T,in_c_k]
        if self.isConv:
            q = self.q_enc_conv(q.permute(0,2,1)).permute(0,2,1)  #[B, T, hid_c]
            k = self.k_enc_conv(k.permute(0,2,1)).permute(0,2,1)  #[B, T, hid_c]
        if self.isTrans:
            if not self.isConv:
                q = self.q_linear(q)  #[B,T,hid_c]
                k = self.k_linear(k)  # [B,T,hid_c]

            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.bool()
            q = self.q_transformer_encoder(q.permute(1,0,2), attn_mask, src_key_padding_mask) #[T, B, hid_c]
            k = self.k_transformer_encoder(k.permute(1,0,2), attn_mask, src_key_padding_mask)  # [T, B, hid_c]
            q = q.permute(1,0,2) #[B, T, c]
            k = k.permute(1, 0, 2)  # [B, T, c]

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool()
            q = q.masked_fill(
                src_key_padding_mask.unsqueeze(2),
                float(0),
            )
            k = k.masked_fill(
                src_key_padding_mask.unsqueeze(2),
                float(0),
            )

        if cal_path:
            dist = self.Euclidean_dist(k, q)  # [B, Tm, Ta]
            path01 = self.dtw_path(dist, real_length)#[B, Tm, Ta]

            path01 = torch.Tensor(path01).cuda()
        else:
            dist = torch.zeros((q.size()[0], q.size(-1), k.size()[-1])).cuda()
            path01 = torch.zeros((q.size()[0], q.size(-1), k.size()[-1])).cuda()

        return path01, dist,  q.permute(0,2,1), k.permute(0,2,1)   #[B, Tm, Ta] [B, Tm, Ta] [B, c, Tm] [B, c, Ta]



class AlignNet_losses(_Loss):
    def __init__(self, useTripletloss=False):
        super(AlignNet_losses, self).__init__()
        self.euc_dist = nn.PairwiseDistance(p=2)  # 欧式距离
        self.useTripletloss = useTripletloss
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss()



    def forward(self, motion_latent, audio_latent, target, real_length):# [B, c, Tm] [B, c, Ta] [B, Tm, Ta] [B]
        B, T, _ = target.size() #[B, Tm, Ta]
        C = motion_latent.size()[1]

        col_index = torch.argmax(target, dim=2)  # [B, Ta]
        row_index = torch.linspace(0, T - 1, T).unsqueeze(0).expand((B, T)).long().cuda()  # [B, Tm]
        rand_index = (row_index + torch.randint_like(row_index, 2, T-1).cuda()) % T  # [B, Tm]

        loss = torch.Tensor([0.0]).cuda()
        for b in range(B):

            anchor = torch.index_select(audio_latent[b], 1, col_index[b]).transpose(1, 0)  #[Ta, c]
            pos = torch.index_select(motion_latent[b], 1, row_index[b]).transpose(1, 0)  #[Tm, c]
            neg = torch.index_select(motion_latent[b], 1, rand_index[b]).transpose(1, 0) #[Tm, c]

            if not self.useTripletloss:
                #mse loss
                pos_dist = self.euc_dist(anchor, pos)  # [T]
                neg_dist = self.euc_dist(anchor, neg)  # [T]
                loss += self.mse_loss(pos_dist, torch.zeros_like(pos_dist).cuda())
                loss += self.mse_loss(neg_dist, torch.ones_like(neg_dist).cuda())
            else:
                #triplet loss
                loss += self.triplet_loss(anchor, pos, neg)

        return loss



class DTWModel(nn.Module):
    def __init__(self, opt):
        super(DTWModel, self).__init__()
        self.phase = opt.phase
        self.batch_size = opt.batch_size
        print("batch size {}".format(self.batch_size))
        self.kernel_size = opt.kernel_size
        self.isnorm = opt.isnorm
        self.isConv = opt.isConv
        self.isTrans = opt.isTrans
        self.total_length = opt.total_length
        print("isnorm: {} isConv: {} isTrans: {} total_length: {}".format(self.isnorm, self.isConv, self.isTrans, self.total_length))
        self.model_dir = os.path.join(opt.net_root, opt.net_path)  # './all_nets/net'
        mkdir(self.model_dir)

        gpu_ids = opt.gpu_ids
        self.device_count = len(gpu_ids)
        print("device_cont:{}".format(self.device_count))

        self.gen = DTWEncTrans(self.kernel_size, 126, 80, 128, self.isnorm, self.isConv, self.isTrans)
        self.gen.apply(self.__weights_init('kaiming'))
        self.gen = nn.DataParallel(self.gen)
        self.gen.cuda()

        gen_params = list(self.gen.parameters())
        self.gen_opt = torch.optim.Adam(
                [p for p in gen_params if p.requires_grad],
                lr=opt.lr_gen, betas=(0.9, 0.999))


        self.corr_w = opt.corr_w
        self.useTripletloss = opt.useTripletloss
        print("useTripletloss:{}.".format(self.useTripletloss))
        self.align_losses = AlignNet_losses(self.useTripletloss)

        self.cal_path = self.phase == 'test'
        print("calpath:{}.".format(self.cal_path))


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

    def make_attn_mask(self, times=1, pad=50):
        negetive_infinity = -100000.0
        attn_mask = negetive_infinity * torch.ones((self.total_length, self.total_length))
        for i in range(self.total_length):
            start = max(0, i - pad)
            finish = min(self.total_length, i + pad)
            attn_mask[i, start: finish] = torch.zeros(finish - start)

        return attn_mask.repeat(times,1)

    def set_input(self, data):
        self.matrix = data["matrix"].clone().detach().float().cuda()   # [B, Tm, Ta]

        self.motion_feature = data['motion_feature'].clone().detach().float().cuda()  # [B, 126, T]
        self.audio_feature = data['audio_feature'].clone().detach().float().cuda() # [B, 80, 263]

        self.attn_mask = self.make_attn_mask(self.device_count).cuda()  # [T, T]
        self.padding_mask = data["padding_mask"].clone().detach().float().cuda() #[B, T]
        self.real_length = data["real_length"].clone().detach().float().cuda() #[1]


    def forward(self, data):
        self.set_input(data)
        #[B, Tm, Ta] [B, Tm, Ta] [B, c, Tm] [B, c, Ta]
        self.corr, self.dist, self.motion_latent, self.audio_latent = self.gen(self.motion_feature,
                                                                                   self.audio_feature,
                                                                                   self.real_length,
                                                                                   self.attn_mask,
                                                                                   self.padding_mask,
                                                                                   cal_path=self.cal_path)

    def backward_G(self):
        # corr loss
        loss_music = self.corr_w * self.align_losses(self.motion_latent, self.audio_latent, self.matrix, self.real_length)
        loss_total = loss_music# + loss_source
        loss_total.backward()

        ret_dict = {
            'corr': loss_music,
        }
        return ret_dict


    def optimize(self):
        self.gen_opt.zero_grad()
        gen_loss_dict = self.backward_G()
        self.__update_dict(self.loss_dict, gen_loss_dict)  # 更新loss字典
        self.gen_opt.step()


    def test(self, data):
        '''For producing results'''
        self.eval()
        self.forward(data)
        # self.train()
        self.cuda()

        out_dict = {
            "target_mat": self.matrix,  # [B, T, T]
            "pred_mat": self.corr,  # [B, T, T]
            "dist": self.dist,

            "real_length": self.real_length #[1]
        }

        return out_dict


    def getlossDict(self):
        return self.loss_dict


    def save_networks(self, epoch):
        gen_name = os.path.join(self.model_dir, 'gen_%08d.pt' % epoch)
        torch.save(self.gen.module.state_dict(), gen_name)

        opt_name = os.path.join(self.model_dir, 'g_optimizer.pt')
        torch.save(self.gen_opt.state_dict(), opt_name)






