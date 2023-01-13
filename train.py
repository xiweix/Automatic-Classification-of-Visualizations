# -*- coding: utf-8 -*-
import os
import time
import gzip
import csv
import warnings

import click
import dataset as database
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

warnings.filterwarnings("ignore")


class BeagleDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_folder_path, csv_file, img_size, train_ok):
        self.folder2_list = []
        self.label_list = []
        self.folder1_list = []
        self.name_list = []
        self.image_name_list = []
        self.img_size = img_size
        self.train_ok = train_ok

        with open(csv_file, 'r') as f:
            prj_info = csv.DictReader(f)
            for row in prj_info:
                if prj_info.line_num != 1:
                    self.label_list.append(row['label1'])
                    self.folder1_list.append(row['folder1'])
                    self.folder2_list.append(row['folder2'])
                    self.name_list.append(
                        os.path.splitext(row['imagename'])[0])
                    self.image_name_list.append(row['imagename'])

        self.imgs_folder_path = imgs_folder_path

    def __len__(self):
        return len(self.folder2_list)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.imgs_folder_path,
            self.folder1_list[idx],
            self.folder2_list[idx],
            self.image_name_list[idx],
        )
        tensor_path = os.path.join(
            self.imgs_folder_path,
            self.folder1_list[idx],
            self.folder2_list[idx],
            f'{self.name_list[idx]}.pt.gz',
        )

        if self.train_ok:
            img = Image.open(img_path).convert('RGB')
            loader = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            img_tensor = loader(img)
        else:
            img_tensor = None
            if os.path.exists(tensor_path):
                with gzip.open(tensor_path, 'rb') as f:
                    img_tensor = torch.load(f, map_location='cpu')
                if img_tensor.size(1) != self.img_size:
                    img_tensor = None
            if img_tensor is None:
                img = Image.open(img_path).convert('RGB')
                loader = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ])
                img_tensor = loader(img)
                with gzip.open(tensor_path, 'wb') as f:
                    torch.save(img_tensor, f)

        label = int(self.label_list[idx])
        label_tensor = torch.as_tensor(label, dtype=torch.float32)
        return (img_tensor, label_tensor)


def train(model_name, model, device, train_loader, optimizer, epoch, loss_batch, outputdir, criterion):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch
        data, target = data.to(device), target.to(device)
        target = target.long()
        optimizer.zero_grad()

        if 'googlenet' in model_name:
            output, output_aux1, output_aux2 = model(data)
            loss0 = criterion(output, target)
            loss1 = criterion(output_aux1, target)
            loss2 = criterion(output_aux2, target)
            loss = (loss0 + loss1 + loss2) / 3

        elif 'inception' in model_name:
            output, output_aux = model(data)
            loss0 = criterion(output, target)
            loss1 = criterion(output_aux, target)
            loss = (loss0 + loss1) / 2

        else:
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())
    with open(os.path.join(outputdir, 'loss_info.txt'), 'a') as ftxt:
        ftxt.write(f'epoch_{epoch}: ')
        ftxt.write(f'train_loss: {loss.item()}\n')
    print(f'Train Epoch: {epoch} ' +
          f'[{(batch_idx+1) * len(data)}/{len(train_loader.dataset)}]\t' +
          f'Loss: {loss.item():.6f}')


def val(model_name, model, device, val_loader, command, num_classes, loss_record, accuracy_record, outputdir, criterion, epoch):
    model.eval()
    val_dir = os.path.join(outputdir, f'{command}_result')
    os.makedirs(val_dir, exist_ok=True)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    accuracy = list(0. for i in range(num_classes + 1))

    output_total = []
    target_total = []
    label_total = []
    predicted_total = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            # target = target.unsqueeze(1).long()
            target = target.long()

            output = model(data)

            predicted = output.argmax(dim=1, keepdim=True)

            for i in range(target.size()[0]):
                label = int(target[i])
                pred = int(predicted[i])

                label_total.append(label)
                predicted_total.append(pred)

                a = (label == pred)
                class_correct[label] += int(a)
                class_total[label] += 1

            output_total.append(output)
            target_total.append(target)
            # print('Label: should be 200: ', len(label_total))
            # print('Prediction: should be 200: ', len(predicted_total))
    correct = 0
    total = 0
    for l in range(num_classes):
        if class_total[l] == 0:
            accuracy[l] = 0
        else:
            correct += int(class_correct[l])
            total += class_total[l]
            accuracy[l] = int(class_correct[l]) / class_total[l] * 100.

    accuracy[num_classes] = correct / total * 100.
    accuracy_record.append(accuracy[num_classes])

    # print('Label: should be 8k: ', len(label_total))
    # print('Prediction: should be 8k: ', len(predicted_total))

    labels_np = np.array(label_total)
    predicted_np = np.array(predicted_total)

    np.save(os.path.join(val_dir, f'Epoch{epoch}_labels.npy'), labels_np)
    np.save(os.path.join(val_dir, f'Epoch{epoch}_predictions.npy'), predicted_np)

    plt.figure(f'Epoch{epoch}_Output')
    plt.title(f'Epoch{epoch}_Output')
    plt.xlabel('Labels')
    plt.ylabel('prdLabels')
    plt.scatter(labels_np, predicted_np)
    plt.savefig(os.path.join(val_dir, f'Epoch{epoch}_Output.png'))
    plt.close()

    output_tensor = torch.cat(output_total)
    target_tensor = torch.cat(target_total)
    val_loss = criterion(output_tensor, target_tensor).item()
    loss_record.append(val_loss)

    if epoch == 1:
        line = []
        line.append('epoch')
        line.append('val_loss')
        line.extend(list(f'accuracy_{i}' for i in range(num_classes + 1)))
        with open(os.path.join(outputdir, f'{command}_accuracy.csv'), 'a') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(line)

    line = []
    line.append(epoch)
    line.append(val_loss)
    line.extend(accuracy)
    with open(os.path.join(outputdir, f'{command}_accuracy.csv'), 'a') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(line)

    with open(os.path.join(outputdir, 'loss_info.txt'), 'a') as ftxt:
        ftxt.write(f'epoch_{epoch}: ')
        ftxt.write(f'{command}_loss: {val_loss}; ')
        ftxt.write(f'{command}_accuracy: {accuracy[num_classes]}\n')

    if 'test' in command:
        with open(os.path.join(outputdir, 'model_info.txt'), 'a') as ftxt:
            ftxt.write(f'------{command} Result------\n')
            ftxt.write(
                f'Epoch: {epoch}\tLoss: {val_loss}\tAccuracy: {accuracy[num_classes]}\n')

    print(f'{command} Epoch: {epoch} ' +
          f'Loss: {val_loss:.6f}\tAccuracy: {accuracy[num_classes]:.4f}%')


def make_outputdir(postfix=''):
    if postfix is None:
        postfix = ''
    timestamp = time.strftime(f'%Y-%m-%d-%H-%M-%S{postfix}')
    outputdir = os.path.join(os.path.dirname(os.getcwd()), 'output',
                             f'{timestamp}')
    os.makedirs(outputdir, exist_ok=True)
    return outputdir


@click.command()
@click.option('--model-name', type=str, default='googlenet', help='model name')
@click.option('--batch-size',
              type=int,
              default=128,
              help='batch size for train (default: 128)')
@click.option('--val-batch-size',
              type=int,
              default=200,
              help='batch size for validation and test (default: 200)')
@click.option('--epochs',
              type=int,
              default=200,
              help='number of epochs (default: 200)')
def main(batch_size, val_batch_size, model_name, epochs):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    print(model_name)
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    criterion = nn.CrossEntropyLoss()

    beagle_dataset_train = BeagleDataset(
        imgs_folder_path=os.path.join(
            os.path.dirname(os.getcwd()),
            'dataset',
        ),
        csv_file=os.path.join(
            os.path.dirname(os.getcwd()),
            'dataset',
            'train.csv',
        ),
        img_size=299 if 'inception' in model_name else 224,
        train_ok=True)

    beagle_dataset_val = BeagleDataset(
        imgs_folder_path=os.path.join(
            os.path.dirname(os.getcwd()),
            'dataset',
        ),
        csv_file=os.path.join(
            os.path.dirname(os.getcwd()),
            'dataset',
            'val.csv',
        ),
        img_size=299 if 'inception' in model_name else 224,
        train_ok=False)

    beagle_dataset_test = BeagleDataset(
        imgs_folder_path=os.path.join(
            os.path.dirname(os.getcwd()),
            'dataset',
        ),
        csv_file=os.path.join(
            os.path.dirname(os.getcwd()),
            'dataset',
            'test.csv',
        ),
        img_size=299 if 'inception' in model_name else 224,
        train_ok=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=beagle_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=beagle_dataset_val,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=beagle_dataset_test,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # outputdir = make_outputdir()
    outputdir = os.path.join(os.path.dirname(os.getcwd()), 'beagle_output',
                             f'random_model.{model_name}-batchsize.{batch_size}')
    os.makedirs(outputdir, exist_ok=True)

    model_dir = os.path.join(outputdir, 'model')
    trainresult_dir = os.path.join(outputdir, 'train_result')
    valresult_dir = os.path.join(outputdir, 'val_result')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(trainresult_dir, exist_ok=True)
    os.makedirs(valresult_dir, exist_ok=True)

    with open(os.path.join(outputdir, 'model_info.txt'), 'a') as ftxt:
        ftxt.write(f'model name: {model_name}\n')
        ftxt.write(f'batch size: {batch_size}\n')
        ftxt.write(f'epochs: {epochs}\n')

    num_classes = 24

    model = getattr(models, model_name)(
        num_classes=num_classes).to(device)
    optimizer = optim.Adam(
        model.parameters(), weight_decay=1e-4, amsgrad=False)

    loss_batch = []
    loss_record = []
    accuracy_record = []

    for epoch in range(1, epochs + 1):
        # train
        train(model_name, model, device, train_loader,
              optimizer, epoch, loss_batch, outputdir, criterion)
        # save train results
        np.save(os.path.join(trainresult_dir, 'epoch%s_loss.npy' %
                             epoch), np.array(loss_batch))

        batch = np.arange(len(np.array(loss_batch))) + 1
        plt.figure('Loss-Batch %s' % epoch)
        plt.title('epoch-%s-Loss' % epoch)
        plt.xlabel('# of batches')
        plt.ylabel('loss')
        plt.plot(batch, np.array(loss_batch))
        plt.savefig(os.path.join(trainresult_dir, 'epoch-%s-Loss.png' % epoch))
        plt.close()

        model_path = os.path.join(model_dir, f'epoch.{epoch}.statedict.pt.gz')
        with gzip.open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)

        # validation
        val(model_name, model, device, val_loader,
            'val', num_classes, loss_record, accuracy_record, outputdir, criterion, epoch)

    # save validation results
    np.save(os.path.join(valresult_dir, 'loss.npy'), np.array(loss_record))
    np.save(os.path.join(valresult_dir, 'accuracy.npy'),
            np.array(accuracy_record))

    min_loss = np.min(loss_record)
    min_loss_epoch = loss_record.index(min_loss) + 1
    min_loss_epoch_accuracy = accuracy_record[min_loss_epoch - 1]

    acc_epoch = np.arange(len(np.array(accuracy_record))) + 1
    plt.figure('Accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(acc_epoch, np.array(accuracy_record))
    plt.savefig(os.path.join(valresult_dir, 'Accuracy.png'))
    plt.close()

    print(
        f'min loss epoch: {min_loss_epoch}; loss: {min_loss}; Accuracy: {min_loss_epoch_accuracy}')

    max_accuracy = np.max(accuracy_record)
    max_accuracy_epoch = accuracy_record.index(max_accuracy) + 1
    max_accuracy_epoch_loss = loss_record[max_accuracy_epoch - 1]
    print(
        f'max accuracy epoch: {max_accuracy_epoch}; accuracy: {max_accuracy}')

    with open(os.path.join(outputdir, 'model_info.txt'), 'a') as ftxt:
        ftxt.write('------Validation Result------\n')
        ftxt.write(
            f'Min Loss Epoch: {min_loss_epoch}\tLoss: {min_loss}\tAccuracy: {min_loss_epoch_accuracy}\n')
        ftxt.write(
            f'Max Accuracy Epoch: {max_accuracy_epoch}\tLoss: {max_accuracy_epoch_loss}\tAccuracy: {max_accuracy}\n')

    # test
    test_model = getattr(models, model_name)(num_classes=num_classes)
    with gzip.open(os.path.join(model_dir, f'epoch.{min_loss_epoch}.statedict.pt.gz'), 'rb') as f:
        test_model.load_state_dict(torch.load(f))
        test_model.to(device)
        val(model_name, test_model, device, test_loader, 'test-minloss',
            num_classes, [], [], outputdir, criterion, min_loss_epoch)

    test_model = getattr(models, model_name)(num_classes=num_classes)
    with gzip.open(os.path.join(model_dir, f'epoch.{max_accuracy_epoch}.statedict.pt.gz'), 'rb') as f:
        test_model.load_state_dict(torch.load(f))
        test_model.to(device)
        val(model_name, test_model, device, test_loader, 'test-maxaccuracy',
            num_classes, [], [], outputdir, criterion, max_accuracy_epoch)


if __name__ == '__main__':
    main()
