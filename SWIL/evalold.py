import torch
import torch.utils.data
from util.parser import get_parser_with_args
from util.helpers import initialize_metrics
import os
from tqdm import tqdm
from util.path_dataloaders import full_test_loader, CDDloader  # 引入CDDloader
import cv2

# 读取命令行参数
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

# 设置设备
dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_path = '/mnt/Disk1/BCD_Dataset/LEVIR-CD/'

# 获取测试数据加载器
test_dataset = full_test_loader(data_path)
test_loader = torch.utils.data.DataLoader(CDDloader(test_dataset, aug=False), batch_size=1, shuffle=False)

# 模型权重路径
weight_path = '/mnt/Disk2/wangzhibin/FMS-MSAA4/weight/LEVIR.pt'

# 输出路径
base_path = '/mnt/Disk2/wangzhibin/FMS-MSAA4/LEVIR/'

# 如果输出路径不存在，则创建
if not os.path.exists(base_path):
    os.mkdir(base_path)

# 加载模型
model = torch.load(weight_path, map_location={'cuda:0': 'cuda:1'})
model.eval()

# 初始化指标
test_metrics = initialize_metrics()

index_img = 0
with torch.no_grad():
    tbar = tqdm(test_loader)
    for sample in tbar:
        batch_img1, batch_img2, labels, paths = sample
        current_path = paths[0]
        current_image_name = test_dataset[index_img]['filename']  # 使用文件名而不是路径
        # 处理图像
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # 获取预测结果
        labels = labels.data.cpu().numpy()
        labels = labels.squeeze() * 255

        cd_preds = model(batch_img1, batch_img2)  # 将batch_img1和batch_img2作为两个独立的参数传递
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        # 保存预测结果图像
        base_name = os.path.splitext(current_image_name)[0]  # 获取没有扩展名的文件名
        file_path = os.path.join(base_path, base_name + '.png')  # 添加_pred.png扩展名
        print(file_path)
        cv2.imwrite(file_path, cd_preds)


        # 仅执行自己的时候保存gt图片
        # file_path1 = os.path.join(base_path, os.path.splitext(current_image_name)[0] + '_gt.png')
        # cv2.imwrite(file_path1, labels)
        # print(file_path1)
        # file_path1 = os.path.join(base_path, base_name + '_gt.png')  # 添加_pred.png扩展名
        # print(file_path)
        # cv2.imwrite(file_path1, labels)
        index_img += 1