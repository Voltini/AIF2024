import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import MNISTNet

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, optimizer, loader, writer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f"training loss: {mean(running_loss)}")
        writer.add_scalar("training loss", mean(running_loss), epoch)


def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name", type=str, default="MNIST", help="Experiment name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

    args = parser.parse_args()
    exp_name = args.exp_name
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    writer = SummaryWriter(f"runs/{exp_name}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        "./data", download=True, train=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        "./data", download=True, train=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    net = MNISTNet()
    net.to(device)

    optimizer = optim.SGD(params=net.parameters(), lr=learning_rate)

    train(net, optimizer, trainloader, writer, epochs)
    test_acc = test(net, testloader)

    # add embeddings to tensorboard
    perm = torch.randperm(len(trainset.data))
    images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
    images = images.unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = net.get_features(images)
        writer.add_embedding(
            embeddings, metadata=labels, label_img=images, global_step=1
        )

    # save networks computational graph in tensorboard
    writer.add_graph(net, images)
    # save a dataset sample in tensorboard
    img_grid = torchvision.utils.make_grid(images[:64])
    writer.add_image("mnist_images", img_grid)
    print(f"Test accuracy: {test_acc}")
    torch.save(net.state_dict(), f"./weights/{exp_name}_net.pth")
