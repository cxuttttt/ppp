import os
import re
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.backbone.AgentNet import AgentAttention
from models.backbone.Mca import MCA_Attention
import timm
from models.backbone.efficientnetv2_timm import Efficientnetv2
# from modelss.recomo.revcol import revcol_base,revcol_large
# from RevColmain.models.revcol import revcol_large
from models.backbone.stvit import stvit_base
from models.backbone.cswin import CSWin_64_12211_tiny_224, CSWin_64_24322_small_224, CSWin_96_24322_base_384, \
    CSWin_96_24322_base_224
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
    
from models.backbone.AgentNet import AgentAttention
from models.block.Base import ChannelChecker
from models.head.FCN import FCNHead
from models.neck.SWE import SWE
from models.neck.FPN import FPNNeck
from collections import OrderedDict
from util.common import ScaleInOutput
from torchvision.models.feature_extraction import create_feature_extractor
from models.modules import FeedForward

from models.block.Base import Conv3Relu
from models.neck.CSF import CSF


class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))  # backbone的名称中必须在"_"之后加上它的通道数
        # print(self.inplanes )
        self.dl = opt.dual_label
        self.auxiliary_head = False    # 9.30经过实验测试，这个没有用

        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)
        self.mca=MCA_Attention(self.inplanes,4)
        self.check_channels = ChannelChecker(self.backbone, self.inplanes, opt.input_size)
        self.Agattention_1=AgentAttention(self.inplanes, 4, True, 0.2, 0.2)
        self.Agattention_2=AgentAttention(self.inplanes*2, 4, True, 0.2, 0.2)
        self.Agattention_3=AgentAttention(self.inplanes*4, 4, True, 0.2, 0.2)
        self.Agattention_4=AgentAttention(self.inplanes*8, 4, True, 0.2, 0.2)
        self.Agattention_1_b=AgentAttention(self.inplanes, 4, True, 0.2, 0.2)
        self.Agattention_2_b=AgentAttention(self.inplanes*2, 4, True, 0.2, 0.2)
        self.Agattention_3_b=AgentAttention(self.inplanes*4, 4, True, 0.2, 0.2)
        self.Agattention_4_b=AgentAttention(self.inplanes*8, 4, True, 0.2, 0.2)
        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   # todo:这里预训练初始化和 hrnet主干网络的初始化有冲突，必须要改！
        self._model_summary(opt.input_size)

        self.feedforward_1 = FeedForward(self.inplanes, 1, 0)
        self.feedforward_2 = FeedForward(self.inplanes * 2, 1,0)
        self.feedforward_3 = FeedForward(self.inplanes * 4, 1, 0)
        self.feedforward_4 = FeedForward(self.inplanes * 8, 1, 0)

        self.swe = SWE(in_ch=self.inplanes * 2, out_ch= self.inplanes)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.stage1 = Conv3Relu(self.inplanes * 3, self.inplanes)
        self.stage2 = Conv3Relu(self.inplanes * 2, self.inplanes)
    
        self.csf = CSF(self.inplanes * 3, self.inplanes * 3)

    def forward(self, xa, xb, tta=False):
        if not tta:  # 如果不用测试时增强
            return self.forward_once(xa, xb)
        else:
            return self.forward_tta(xa, xb)

    def forward_once(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.backbone(xa)  # torch.Size([4, 64, 112, 112]),torch.Size([4, 128, 56, 56]),torch.Size([4, 256, 28, 28]),torch.Size([4, 512, 14, 14])
        fa1, fa2, fa3, fa4 = self.check_channels(fa1, fa2, fa3, fa4)#torch.Size([4, 40, 112, 112]),torch.Size([4, 80, 56, 56]),torch.Size([4, 160, 28, 28]),torch.Size([4, 320, 14, 14])

        fb1, fb2, fb3, fb4 = self.backbone(xb)
        fb1, fb2, fb3, fb4 = self.check_channels(fb1, fb2, fb3, fb4)

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4   # 多尺度特征

        change1, change2 = self.neck(ms_feats)  # torch.Size([4, 40, 112, 112]),torch.Size([4, 80, 56, 56])

        yL,yH,local =self.swe(change2)   # torch.Size([4, 40, 28, 28])
 
        changeall=torch.cat([yL,yH,local], 1)  #torch.Size([4, 120, 28, 28])

        changeall = self.csf(changeall)    #torch.Size([4, 120, 28, 28])

        change1_all = self.stage1(self.up(changeall))  #torch.Size([4, 40, 112, 112])

        out = self.stage2(torch.cat([change1, change1_all], 1))   #torch.Size([4, 40, 112, 112])
        
        # change2 = change2.squeeze(0)

        # output_dir = '/mnt/Disk2/xutianhao/FMS-MSAA4/local'
        # os.makedirs(output_dir, exist_ok=True)

        # # 遍历每个样本的每个通道
        # for i in range(local.size(0)):
        # # 获取当前样本的通道数
        #     num_channels = local[i].size(0)
        #     print(f"Sample {i} has {num_channels} channels.")
        #     for j in range(num_channels):
        #     # 提取第 j 个通道的数据，并增加两个维度以符合 save_image 的要求
        #         channel_data = local[i, j].unsqueeze(0).unsqueeze(0)
        #     # 保存每个通道的图像
        #         save_image(channel_data, os.path.join(output_dir, f'sample_{i}_channel_{j}.png'))

        # out = self.head_forward(ms_feats, out, out_size=(h_input, w_input))  #torch.Size([4, 2, 448, 448])

        return out

    def forward_tta(self, xa, xb):
        bs, c, h, w = xa.shape
        # mutil_scales = [1.0, 0.8, 1.2, 0.5, 2]
        # mutil_scales = [1.25, 1.0, 0.7]
        mutil_scales = [1.0, 0.834, 0.667, 0.542]

        out1, out2 = 0, 0
        for single_scale in mutil_scales:    # 多尺度测试
            single_scale = (int((h * single_scale) / 32) * 32, int((w * single_scale) / 32) * 32)
            xa_size = F.interpolate(xa, single_scale, mode='bilinear', align_corners=True)
            xb_size = F.interpolate(xb, single_scale, mode='bilinear', align_corners=True)

            out_1 = self.forward_once(xa_size, xb_size)  # 正常forward


            if self.dl:  # 双标签
                out1_1, out1_2 = out_1[0], out_1[1]


            else:   # 单标签
                out1_1 = out_1

                out1 += F.interpolate(out1_1,
                                      size=(h, w), mode='bilinear', align_corners=True)

        return (out1, out2) if self.dl else out1

    def head_forward(self, ms_feats, change, out_size):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats

        out1 = F.interpolate(self.head1(change), size=out_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(change), size=out_size,
                             mode='bilinear', align_corners=True) if self.dl else None

        if self.training and self.auxiliary_head:
            aux_stage1_out1 = F.interpolate(self.aux_stage1_head1(torch.cat([fa1, fb1], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage1_out2 = F.interpolate(self.aux_stage1_head2(torch.cat([fa1, fb1], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage2_out1 = F.interpolate(self.aux_stage2_head1(torch.cat([fa2, fb2], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage2_out2 = F.interpolate(self.aux_stage2_head2(torch.cat([fa2, fb2], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage3_out1 = F.interpolate(self.aux_stage3_head1(torch.cat([fa3, fb3], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage3_out2 = F.interpolate(self.aux_stage3_head2(torch.cat([fa3, fb3], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage4_out1 = F.interpolate(self.aux_stage4_head1(torch.cat([fa4, fb4], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage4_out2 = F.interpolate(self.aux_stage4_head2(torch.cat([fa4, fb4], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            return (out1, out2,
                    aux_stage1_out1, aux_stage1_out2, aux_stage2_out1, aux_stage2_out2,
                    aux_stage3_out1, aux_stage3_out2, aux_stage4_out1, aux_stage4_out2) \
                if self.dl else (out1, aux_stage1_out1, aux_stage2_out1, aux_stage3_out1, aux_stage4_out1)
        else:
            return (out1, out2) if self.dl else out1

    def _init_weight(self, pretrain=''):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 只要是卷积都操作，都对weight和bias进行kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  # bn层都权重初始化为1， bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def _model_summary(self, input_size):
        input_sample = torch.randn(1, 3, input_size, input_size)  # for cswin


    def _create_backbone(self, backbone):
        if 'efficientnet' in backbone:
            self.backbone = Efficientnetv2(backbone)
        elif 'cswin' in backbone:
            if '_t_' in backbone:
                self.backbone = CSWin_64_12211_tiny_224(pretrained=True)
            elif '_s_' in backbone:
                self.backbone = CSWin_64_24322_small_224(pretrained=True)
            elif '_b_' in backbone:
                self.backbone = CSWin_96_24322_base_384(pretrained=True)
            elif '_b448_' in backbone:
                self.backbone = CSWin_96_24322_base_224(pretrained=True)
        elif 'stvit' in backbone:
            self.backbone1=stvit_base()
            self.backbone = stvit_base()
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(self.inplanes, neck)
    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 2)

    def _create_heads(self, head):
        self.head1 = self._select_head(head)
        self.head2 = self._select_head(head) if self.dl else None

        if self.auxiliary_head:   # 辅助分支仅使用简单的FCN
            self.aux_stage1_head1 = FCNHead(self.inplanes * 2, 2)
            self.aux_stage1_head2 = FCNHead(self.inplanes * 2, 2) if self.dl else None

            self.aux_stage2_head1 = FCNHead(self.inplanes * 4, 2)
            self.aux_stage2_head2 = FCNHead(self.inplanes * 4, 2) if self.dl else None

            self.aux_stage3_head1 = FCNHead(self.inplanes * 8, 2)
            self.aux_stage3_head2 = FCNHead(self.inplanes * 8, 2) if self.dl else None

            self.aux_stage4_head1 = FCNHead(self.inplanes * 16, 2)
            self.aux_stage4_head2 = FCNHead(self.inplanes * 16, 2) if self.dl else None


class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        """
        :param ckp_paths:
        :param device:
        :param method: 模型集成的可选方式；经过测试，avg2效果最好
                       avg1: 模型输出后，不经过 softmax ，直接求平均，然后再二值化
                       avg2: 模型输出后，先经过 softmax ，再求平均，最后二值化
                       vote: 模型输出后， 先二值化，然后进行投票，and、or、多数胜少数
        """
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        print("-"*50+"\n--Ensamble method: {}".format(method))
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            # model = torch.load(ckp_path)["shadow3"]
            # print("shadows3")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb, tta=False):
        """
        :param xa:
        :param xb:
        :param tta:
        :return: 经过二值化的结果
        """
        xa, xb = self.scale.scale_input((xa, xb))
        out1, out2 = 0, 0
        cd_pred1, cd_pred2 = None, None

        for i, model in enumerate(self.models_list):
            outs,att = model(xa, xb, tta)
            # print('======ense',outs.shape)
            # outs = F.interpolate(outs, size=(512, 512), mode='bilinear', align_corners=True)
            if not isinstance(outs, tuple):  # 如果cd_pred是tuple, 就表明它是双向预测输出; 否则就把单预测复制一份，变成双预测
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))  # 先进行softmax
                # print('outs[0]',outs[0].shape)
                out1 += outs[0]
                out2 += outs[1]
                _, cd_pred1 = torch.max(out1, 1)  # 二值化预测结果
                _, cd_pred2 = torch.max(out2, 1)
                print('======cd_pred1',cd_pred1.shape)
                # cd_pred1 = out1[:, 1, ...] > 0.45 * len(self.models_list)    # 调阈值
                # cd_pred2 = out2[:, 1, ...] > 0.45 * len(self.models_list)
            elif self.method == "vote":  # 多数胜少数，当对前景与背景的投票数一样时，认为其为前景
                _, out1_tmp = torch.max(outs[0], 1)  # 二值化预测结果
                _, out2_tmp = torch.max(outs[1], 1)
                out1 += out1_tmp
                out2 += out2_tmp
                cd_pred1 = out1 / i >= 0.5
                cd_pred2 = out2 / i >= 0.5

        if self.models_list[0].dl:
            return cd_pred1, cd_pred2
        else:
            return cd_pred1,att
        


class ModelEMA:

    def __init__(self, model, decay=0.96):
        self.shadow1 = deepcopy(model.module if self.is_parallel(model) else model).eval()
        self.decay = decay
        for p in self.shadow1.parameters():
            p.requires_grad_(False)

        self.shadow2 = deepcopy(self.shadow1)
        self.shadow3 = deepcopy(self.shadow1)   # 第0代的权重设为0
        self.update_count = 0

    def update(self, model):
        with torch.no_grad():
            msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
            for k, v in self.shadow1.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1. - self.decay) * msd[k].detach()   # += 是本地操作，其值原地修改
            for k, v in self.shadow2.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.95
                    v += (1. - 0.95) * msd[k].detach()   # += 是本地操作，其值原地修改
            for k, v in self.shadow3.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.94
                    v += (1. - 0.94) * msd[k].detach()   # += 是本地操作，其值原地修改
        self.update_count += 1

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelSWA:

    def __init__(self, total_epoch=300):
        self.update_count = 0
        self.epoch_threshold = int(total_epoch * 0.8)    # 训练的过程中，前80%的epoch不进行SWA，从80%处开始SWA
        self.swa_model = None

    def update(self, model):
        if self.update_count >= self.epoch_threshold:
            with torch.no_grad():
                if self.swa_model is None:
                    self.swa_model = deepcopy(model.module) if self.is_parallel(model) else deepcopy(model)
                else:
                    msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
                    for k, v in self.swa_model.state_dict().items():
                        if v.dtype.is_floating_point:
                            v *= (self.update_count - self.epoch_threshold)
                            v += msd[k].detach()
                            v /= (self.update_count - self.epoch_threshold + 1)
        self.update_count += 1

    def save(self, swa_ckp_dir_path):
        if self.update_count >= self.epoch_threshold:
            swa_file_path = os.path.join(swa_ckp_dir_path, "swa_{}_{}.pt".format(
                self.update_count-1, self.update_count-self.epoch_threshold))
            torch.save(self.swa_model, swa_file_path)

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Change Detection train')
    # 配置模型

    parser.add_argument("--backbone", type=str, default="cswin_t_64")
    # parser.add_argument("--neck", type=str, default="fpn+aspp+drop")
    # parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")

    parser.add_argument("--pretrain", type=str,
                        # default="./runs/train/37/L.pt")  # 预训练权重路径
                        default="")  # 预训练权重路径main_app.py

    parser.add_argument("--cuda", default='2', help='whether use CUDA')
    parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/liyemei/change_detection/"
                                                           "Mei_CDNet/CDData/LEVIR-CD/")
    # parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/liyemei/change_detection/Mei_CDNet/CDData/MatchCD/test02/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--input-size", type=int, default=448)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=False)

    opt = parser.parse_args()
    model = ChangeDetection(opt).cuda()
    in1 = torch.randn(2, 3, 448, 448).cuda()
    in2 = torch.randn(2, 3, 448, 448).cuda()
    edg = model(in1, in2)
    print(edg.shape)