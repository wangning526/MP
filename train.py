import os
import math
import vit_args
import torch
import torch.optim as optim
from torchvision import transforms
from mpdataset import MPDataset
import vit_model
from utils import train_one_epoch, evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


#加载参数
vitargs = vit_args.get_args()
u_path  = vitargs.u_data_path
v_path  = vitargs.v_data_path
t_path  = vitargs.t_data_path
mp_path = vitargs.m_data_path
batch_size = vitargs.batch_size
epochs = vitargs.epochs
learning_rate = vitargs.lr

# 将数据集划分为训练集和测试集
dataset = MPDataset(u_path, v_path, t_path, mp_path)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
# 创建 DataLoader 对象
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,pin_memory=True)


# 初始化模型、损失函数和优化器
device = torch.device(vitargs.device if torch.cuda.is_available() else "cpu")
model = vit_model().to(device)
criterion = nn.CrossEntropyLoss()

tb_writer = SummaryWriter()

if os.path.exists("./weights") is False:
        os.makedirs("./weights")
if vitargs.weights != "":
    assert os.path.exists(vitargs.weights), "weights file: '{}' not exist.".format(vitargs.weights)
    weights_dict = torch.load(vitargs.weights, map_location=device)
    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))

if vitargs.freeze_layers:
    for name, para in model.named_parameters():
        # 除head, pre_logits外，其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=learning_rate, momentum=0.9, weight_decay=5E-5)
lf = lambda x: ((1 + math.cos(x * math.pi / vitargs.epochs)) / 2) * (1 - vitargs.lrf) + vitargs.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


for epoch in range(epochs):
    # train
    train_loss = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch)

    scheduler.step()

    # validate
    val_loss = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=epoch)
    tags = ["train_loss", "val_loss", "learning_rate"]
    tb_writer.add_scalar(tags[0], train_loss, epoch)
    tb_writer.add_scalar(tags[1], val_loss, epoch)
    tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


