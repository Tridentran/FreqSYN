import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from sklearn.metrics import f1_score

from Code.Utils.Utils import denorm
import argparse



def data_load(name, root_path, pt=True):
    # the data loading function should be implemented by the user
    # the data should be loaded in the following format:
    # name = 'train', 'test' -> the name of the dataset, trainset & testset should be loaded separately
    # root_path -> the path where the data is stored
    data = None
    label = None
    if pt:
        data = torch.tensor(data)
        label = torch.tensor(label)
    return data, label


def data_load_aug(name, root_path, pt=True, aug_path="/path/to/your/augment/data"):
    # first call data_load() to load the original data
    data, label = data_load(name, root_path, pt=False)
    # then call data_load() again to load the augmented data
    aug_data, aug_label = data_load(name, aug_path, pt=False)
    # combine the original data and augmented data
    data = np.concatenate([data, aug_data])
    label = np.concatenate([label, aug_label])

    if pt:
        data = torch.tensor(data)
        label = torch.tensor(label)

    return data, label






def evaluate(model, test_dataloader, device):
    model.to(device)
    model.eval()
    accs = 0.0

    label_save = []
    pred_save = []

    with torch.no_grad():
        for (data, label) in tqdm(test_dataloader):
            data = data.to(device).float()
            label = label.to(device).float()

            label = torch.argmax(label, dim=-1)
            label_save.append(label.cpu().detach().numpy())
            pred = model(data)
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=-1)
            pred_save.append(pred.cpu().detach().numpy())

            correct = torch.eq(pred, label)
            acc = correct.sum().float().item() / len(label)
            accs += acc
        accs /= len(test_dataloader)

    label_save = np.concatenate(label_save)
    pred_save = np.concatenate(pred_save)
    f1_score_value = f1_score(label_save, pred_save, )

    return accs, f1_score_value


def load_config(json_data, cfg_class):
    return cfg_class(**json_data)


