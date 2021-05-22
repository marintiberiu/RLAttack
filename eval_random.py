import csv
import random
from queue import LifoQueue

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    test_dataset = CIFAR10('data', train=False, transform=transform, download=False)
    test_dataset = Subset(test_dataset, range(9000, 9100))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3)

    net = resnet18(pretrained=False, num_classes=10).eval().cuda()
    net.load_state_dict(torch.load('saves/resnet18_cifar10.sv'))

    max_queries = 200
    n_tries = 100
    n_images = 100

    f = open("results/random.csv", 'w', newline='')
    csv_w = csv.writer(f)
    csv_w.writerow(("N Correct", "Queries"))

    rez_list = []

    for idx, data in enumerate(tqdm(test_dataloader, total=n_images)):
        if idx == n_images:
            break

        inputs = data[0].cuda()
        labels = data[1].cuda()

        n_correct = 0
        n_queries = 0

        pixels = LifoQueue()

        for t in range(n_tries):
            mask = torch.zeros_like(inputs)
            for i in range(max_queries):
                x = random.randrange(0, 32)
                y = random.randrange(0, 32)

                pixels.put((x, y))
                if pixels.qsize() > 50:
                    rx, ry = pixels.get()
                    mask[:, :, x, y] = 0

                mask[:, :, x, y] = 1

                out = net((inputs + mask).clamp(-1, 1)).argmax(dim=1)
                if out != labels:
                    break

            n_queries += i + 1
            if i != max_queries:
                n_correct += 1

        csv_w.writerow((n_correct, n_queries))
        f.flush()

    f.close()