import sys
import torch
from tqdm import tqdm


@torch.no_grad()
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input  = data[:-1]
        output = data[-1].reshape(-1)

        pred = model(input.to(device))
        loss = loss_function(pred, output.to(device))
        loss.backward()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        input  = data[:-1]
        output = data[-1].reshape(-1)

        pred = model(input.to(device))
        loss = loss_function(pred, output.to(device))

    return loss