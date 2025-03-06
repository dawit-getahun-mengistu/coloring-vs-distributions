import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cnn import LeNetBasedModel
from cfg import device, DATASET_PATH
from load import FashionMNIST, MNIST


# dataset = FashionMNIST()
dataset = MNIST()
train_loader = dataset.train_loader
test_loader = dataset.test_loader


model = LeNetBasedModel().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    """Training function"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    """Testing function"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return {
        'loss': test_loss,
        'acc': accuracy
    }


def test_with_confusion(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy, np.array(all_preds), np.array(all_labels)


if __name__ == '__main__':
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)

    print(f"Test set result")
    test(model, device, test_loader)
