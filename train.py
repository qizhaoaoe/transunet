import torch
from torch import nn, optim
from TransUnet import TransUnet
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
from dataset import MyDataset, save_img
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, criterion, optimizer, dataloader, save_dir, num_epochs=20):
    dt_size = len(dataloader.dataset)
    all_step = (dt_size-1)//dataloader.batch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        epoch_loss = 0
        step = 0
        for x, y in dataloader:
            step += 1
            inputs = x.type(torch.FloatTensor).to(device)
            labels = y.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            print("%d/%d, train_loss:%0.3f" % (step, all_step, loss.item()))
        torch.save(model, os.path.join(save_dir, f'pytorch_model_{epoch}.pth'))


def test(args, save_dir):
    model = torch.load(args['ckpt'])
    dataset = MyDataset(args['data_dir'], args['data_path_file'], args['input_shape'])
    dataloaders = DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloaders):
            x = x.type(torch.FloatTensor).to(device)
            y_pred = model(x).cpu()
            y_pred = torch.argmax(y_pred, dim=1).numpy().astype(np.uint8)
            save_img(y_pred, os.path.join(save_dir, f'pred{i}.nii.gz'))


def train(args, save_model_dir):
    epochs = args.get('epochs', 100)
    model = TransUnet(args)
    model = model.cuda()
    weights = args.get('weights', [1.0, 100.0])
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters())
    dataset = MyDataset(args['data_dir'], args['data_path_file'], args['input_shape'])
    dataloaders = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, save_model_dir, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='train config file path')
    parser.add_argument('--action', type=str, help='train or test')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f'config path not exists: ', {args.path})
    with open(args.path, 'rb') as f:
        configs = yaml.safe_load_all(f)
        cfs = list(configs)[0]
    print(cfs)
    if args.action == 'train':
        save_model_dir = cfs.get('ckpt', 'models')
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)
        train(cfs, save_model_dir)

    if args.action == 'test':
        test(args)

