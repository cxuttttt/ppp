import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data

from main_model import EnsembleModel
from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
running_metrics =  RunningMetrics(2)
from matplotlib import pyplot as plt
import torch.nn.functional as F

def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()   # 打印GPU信息
    save_path, result_save_path = check_eval_dirs()

    # 准备写日志
    save_results = SaveResult(result_save_path)
    save_results.prepare()

    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)
    #model = ChangeDetection(opt.ckp_paths, device, input_size=opt.input_size).cuda()
    # print(model.models_list)
    if model.models_list[0].head2 is None:
        opt.dual_label = False
    else:
        opt.dual_label = True
    eval_loader = get_eval_loaders(opt)

    p, r, f1, miou, oa, avg_loss = eval_for_metric(model, eval_loader, tta=opt.tta)

    # 写日志
    save_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=512):   # 评估,得到指标(metric)
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    tn_fp_fn_tp = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, name) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            name = str(name)
            name = name[2:]
            name = name[:-3]
            print(name)

            if name =='33.png':

                if not os.path.exists('att01/'+name+'/'):
                    os.mkdir('att01/'+str(name)+'/')

                if criterion is not None:
                    batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

                outs,att = model(batch_img1, batch_img2, tta)
                # att = model(batch_img1, batch_img2, tta)
                # att = att.unsqueeze(0)
                print(att.shape)
                
                outs = att

                if not isinstance(outs, tuple):
                    outs = (outs, outs)
                labels = (batch_label1, batch_label2)

                if criterion is not None:
                    outs = scale.scale_output(outs)
                    val_loss = criterion(outs, labels)  # ((B,2 H,W), (B,2 H,W)) and ((B,H,W), (B,H,W))
                    _, cd_pred1 = torch.max(outs[0], 1)   # 二值化预测结果
                    _, cd_pred2 = torch.max(outs[1], 1)
                else:
                    cd_pred1 = outs[0]
                    cd_pred2 = outs[1]

                cd_preds = (cd_pred1, cd_pred2)

                count = 0
                for j, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
                    

                    if(count == 1):
                        count = 0
                        continue

                    if(count == 0):
                        # att = F.interpolate(att, (256,256), mode='bilinear')
                        # att = att.data.cpu().numpy()
                        # for i in range(33):
                        #     print(i)
                        #     # plt.imshow(att[0][i], cmap=plt.cm.get_cmap('jet'))
                        #     plt.imsave('att01/'+str(name)+'/'+str(i)+'.png',att[0][i], cmap=plt.cm.get_cmap('jet'))
                        #     plt.pause(0.1)
                        #     i+=1
                        att = F.interpolate(att, (256,256), mode='bilinear')
                        att = att.data.cpu().numpy()

                        # 检查 att 的形状
                        print("att shape:", att.shape)

                        # 确保 att 的第一个维度大小大于0
                        if att.shape[0] > 0:
                            for i in range(att.shape[0]):  # 使用 att.shape[0] 作为循环次数
                                plt.imsave('att01/'+str(name)+'/'+str(i)+'.png', att[0][i], cmap=plt.cm.get_cmap('jet'))
                        else:
                            print("Error: att does not contain enough elements.")
                    count += 1
            


    return 1, 2, 3, 4, 5, 6


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection eval')

    # 配置测试参数
    parser.add_argument("--ckp-paths", type=str,
                        default=[
                            "/mnt/Disk2/wangzhibin/FMS/weight/WHU.pt",
                            # "./runs/train/24/best_ckp/",
                            # "./runs/train/35/best_ckp/",
                        ])

    parser.add_argument("--cuda", type=str, default="0")  # GPU编号
    parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/BCD_Dataset/T640_CDData/WHU-CD/")
    # parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk2/liyemei/change_detection/Mei_CDNet/CDData/SYSU-CD/")
    # parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk2/liyemei/change_detection/Mei_CDNet/CDData/WHU-CD/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--tta", type=bool, default=False)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
