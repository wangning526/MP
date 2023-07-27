import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import datetime as dt
import os
import torch
import vit_args
class MPDataset(Dataset):
    def __init__(self, u_path, v_path, t_path, mp_path):
        self.u_path = u_path
        self.v_path = v_path
        self.t_path = t_path
        self.mp_path = mp_path

        self.data = []

        start_date = dt.date(2017, 6, 1)
        end_date = dt.date(2018, 5, 31)
        current_date = start_date
        while current_date <= end_date:
            time = current_date.strftime('%Y%m%d')
            u_file = 'u10m_' + time + '.csv'
            u_file = os.path.join(self.u_path, u_file)
            u_data = np.loadtxt(u_file, delimiter=',')
            v_file = 'v10m_' + time + '.csv'
            v_file = os.path.join(self.v_path, v_file)
            v_data = np.loadtxt(v_file, delimiter=',')
            t_file = 'tmps_' + time + '.csv'
            t_file = os.path.join(self.t_path, t_file)
            t_data = np.loadtxt(t_file, delimiter=',')
            mp_file = 'mp_' + time + '.csv'
            mp_file = os.path.join(self.mp_path, mp_file)
            mp_data = np.loadtxt(mp_file, delimiter=',')
            day = np.array([u_data, v_data, t_data, mp_data])
            self.data.append(day)
            current_date += dt.timedelta(days=1)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get data for the specified index
        data_entry = self.data[index]

        # Separate input and output
        inputs = data_entry[:-1]  # First three channels as input
        output = data_entry[-1]   # Last channel as output

        # Convert input and output to PyTorch tensors
        inputs = torch.FloatTensor(inputs)
        output = torch.FloatTensor(output)

        return inputs, output

