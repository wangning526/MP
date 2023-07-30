from vit_model import vit_base_patch15_75_360 as vit_model
import torch
from torchvision import transforms
import datetime as dt
import vit_args
import os
import numpy as np
import pandas as pd
from utils import gen_grid
from utils import gen_polt
from mpdataset import MPDataset
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#加载参数
vitargs = vit_args.get_args()
u_path  = vitargs.u_data_path
v_path  = vitargs.v_data_path
t_path  = vitargs.t_data_path
mp_path = vitargs.m_data_path
batch_size =  vitargs.batch_size
print('==> Building model..')
model = vit_model()
model.load_state_dict(torch.load('weights\\model.pth'))
model.eval()
spath = 'result'
plot_path = 'plot'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
if not os.path.exists(spath):
    os.makedirs(spath)
#加载数据
print('==> Loading data...')
data = []
start_date = dt.date(2018, 5, 2)
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
    input = np.array([u_data, v_data, t_data, mp_data])
    data.append(input)
    current_date += dt.timedelta(days=1)


dataset = MPDataset(data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
current_date = start_date
print('==> Start testing..')
for batch_idx, data in enumerate(data_loader):
    input = data[:, :3, :, :]
    input = torch.tensor(input, dtype=torch.float32)
    input = transform(input)
    print(input.shape)
    output = model(input)
    # 归一化的范围
    min_val = 0
    max_val = 7.0e+12

    # 反归一化
    output = output * (max_val - min_val) + min_val
    output = output.detach().numpy()
    for i in range(output.shape[0]):
        df = pd.DataFrame(output[i])
        savename = current_date.strftime('%Y%m%d') + '.csv'
        savename = os.path.join(spath, savename)
        gen_grid(df, savename)
        current_date += dt.timedelta(days=1)















