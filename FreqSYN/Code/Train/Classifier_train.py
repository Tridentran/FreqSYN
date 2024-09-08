import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
import sys
import os
import argparse

working_dir = os.getcwd()
sys.path.append(working_dir)

from Code.Module.Classifier.MMCNN import MMCNNModel
from Code.Module.EEG_Encoder.TF_Encoder import Classifier
from Code.Utils.Func import data_load, evaluate, data_load_aug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='BCI_2b', help="dataset to use")
    parser.add_argument("--train_mode", 
                        type=str, 
                        default='normal', 
                        help="normal for training with orignal dataset, aug for training with augmented dataset")  

    args = parser.parse_args()
    dataset = args.dataset
    aug = args.train_mode == 'aug'

    BATCH_SIZE = 1000
    EPOCHS = 500 if not aug else 700

    # device
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset == 'your_dataset':
        root_path = '/path/to/your/data/'
    else:
        raise ValueError(f'Error dataset {dataset}')

    print(f'Using data: {dataset}')
    print(f'Loading data from: {root_path}')
    # data loading
    if aug:
        train_pure_eeg, _, train_label_eeg = data_load_aug(name='train', root_path=root_path, aug_path='/path/to/your/augment/data')
    else:
        train_pure_eeg, _, train_label_eeg = data_load(name='train', root_path=root_path)
    test_pure_eeg, _, test_label_eeg = data_load(name='test', root_path=root_path)

    train_torch_dataset = Data.TensorDataset(train_pure_eeg, train_label_eeg)
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_torch_dataset = Data.TensorDataset(test_pure_eeg, test_label_eeg)
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MMCNNModel.MMCNNModel().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    model_name = 'MMCNN'

    max_acc = 0.5
    best_epoch = 0
    last_f1 = 0.5

    for epoch in range(EPOCHS):
        # model train
        model.train()
        for (train_data, train_label) in tqdm(train_loader):
            optimizer.zero_grad()

            train_data = train_data.float().to(device)
            train_label = train_label.float().to(device)

            predict = model(train_data)
            loss = criterion(predict, train_label)

            loss.backward()
            optimizer.step()
            model.zero_grad()

        test_acc, test_f1 = evaluate(model, test_loader, device)

        tqdm.write(f'epoch:{epoch + 1}, train_loss:{loss}, test_acc:{test_acc:.3f}, test_f1:{test_f1:.3f}')
        
        if not aug:
            save_path = os.path.join(working_dir, f"/Model/{dataset}/{model_name}")
        else:
            save_path = os.path.join(working_dir, f"/Model/{dataset}/rl_aug/{model_name}/")
         
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        if test_acc >= max_acc:
            if os.path.exists(os.path.join(save_path, f'zdm_{best_epoch}_{max_acc:.3f}_{last_f1:.3f}.pkl')):
                os.remove(os.path.join(save_path, f'zdm_{best_epoch}_{max_acc:.3f}_{last_f1:.3f}.pkl'))    # del old model
            print('save model')
            max_acc = test_acc
            best_epoch = epoch + 1
            last_f1 = test_f1
            torch.save(model.state_dict(), os.path.join(save_path, f'zdm_{epoch + 1}_{max_acc:.3f}_{last_f1:.3f}.pkl'))


if __name__ == '__main__':
    main()