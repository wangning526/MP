import os
import math
import sys
import vit_args
import torch
import torch.optim as optim
from torchvision import transforms
from mpdataset import MPDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from vit_model import vit_base_patch15_75_360 as vit_model
from matplotlib import pyplot as plt
import datetime as dt
import numpy as np
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
#加载数据
data = []
start_date = dt.date(2017, 6, 1)
end_date = dt.date(2018, 5, 31)
current_date = start_date
while current_date <= end_date:
    time = current_date.strftime('%Y%m%d')
    u_file = 'u10m_' + time + '.csv'
    u_file = os.path.join(u_path, u_file)
    u_data = np.loadtxt(u_file, delimiter=',')
    v_file = 'v10m_' + time + '.csv'
    v_file = os.path.join(v_path, v_file)
    v_data = np.loadtxt(v_file, delimiter=',')
    t_file = 'tmps_' + time + '.csv'
    t_file = os.path.join(t_path, t_file)
    t_data = np.loadtxt(t_file, delimiter=',')
    mp_file = 'mp_' + time + '.csv'
    mp_file = os.path.join(mp_path, mp_file)
    mp_data = np.loadtxt(mp_file, delimiter=',')
    day = np.array([u_data, v_data, t_data, mp_data])
    data.append(day)
    current_date += dt.timedelta(days=1)
data = np.array(data)


# 将数据集划分为训练集和测试集
dataset = MPDataset(data)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
# 创建 DataLoader 对象
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,pin_memory=True)


# 初始化模型、损失函数和优化器
device = torch.device(vitargs.device if torch.cuda.is_available() else "cpu")
model = vit_model().to(device)
criterion = nn.MSELoss()
model.train()


pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=learning_rate, momentum=0.9, weight_decay=5E-5)
lf = lambda x: ((1 + math.cos(x * math.pi / vitargs.epochs)) / 2) * (1 - vitargs.lrf) + vitargs.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
optimizer.zero_grad()

train_losses = []
val_losses = []

for epoch in range(epochs):
    # train
    loss = torch.zeros(1)
    for batch_idx, data in enumerate(train_loader):
        input = data[:, :3, :, :]  # 取前三个通道作为输入，形状为 (5, 3, 75, 360)
        output = data[:, 3:4, :, :]  # 取第四个通道作为输出，形状为 (5, 1, 75, 360)
        output = output.view(batch_size, 27000)
        input = input.to(torch.float32)
        output = output.to(torch.long)
        pred = model(input.to(device))
        print(pred.shape,output.shape)
        loss = criterion(pred, output.to(device))
        loss.backward()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        loss += loss.item()
    train_loss = loss / len(train_loader)
    train_losses.append(train_loss)
    scheduler.step(train_loss)

    print(
        f'Processing: [{epoch} / {vitargs.epochs}] | Loss: {round((train_loss / len(train_loader)).item(), 6)} | Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')

# validate
val_loss = torch.zeros(1)
with torch.no_grad():
    for batch_idx, data in enumerate(train_loader):
        input = data[:, :3, :, :]  # 取前三个通道作为输入，形状为 (5, 3, 75, 360)
        output = data[:, 3:4, :, :]  # 取第四个通道作为输出，形状为 (5, 1, 75, 360)
        output = output.view(batch_size, 27000)
        input = input.to(torch.float32)
        output = output.to(torch.long)
        pred = model(input.to(device))
        loss = criterion(pred, output.to(device))
        val_loss.append(loss.item())
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss.item())
    print(f'Test Loss: {round((val_loss/len(val_loader)).item(), 6)}')
if os.path.exists("weights") is False:
    os.makedirs("weights")
torch.save(model.state_dict(), "./weights/model.pth")
# 绘制损失函数图像
plt.plot(range(1, vitargs.epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, vitargs.epochs + 1, 10), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Val Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()