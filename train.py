import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import vgg11
from tqdm import tqdm

if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    train_dataset = CIFAR10('data', train=True, transform=transform, download=False)
    test_dataset = CIFAR10('data', train=False, transform=transform, download=False)

    max_epochs = 100
    batch_size = 64
    net = vgg11(pretrained=False, num_classes=10).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=1e-10)
    loss_f = CrossEntropyLoss()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    for epoch in range(max_epochs):
        train_acc = 0
        test_acc = 0

        net.train()
        for idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs = data[0].cuda()
            labels = data[1].cuda()
            outputs = net(inputs)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            train_acc += (outputs.argmax(dim=1) == labels).float().mean()

        scheduler.step()

        with torch.no_grad():
            net.eval()
            for idx, data in tqdm(enumerate(test_dataloader)):
                optimizer.zero_grad()
                inputs = data[0].cuda()
                labels = data[1].cuda()
                outputs = net(inputs)
                test_acc += (outputs.argmax(dim=1) == labels).float().mean()

        train_acc = train_acc / len(train_dataloader) * 100
        test_acc = test_acc / len(test_dataloader) * 100

        print("Epoch", epoch, "train", train_acc, "test", test_acc)

        torch.save(net.state_dict(), 'saves/vgg11_cifar10.sv')