import csv
import random
from queue import Queue

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, vgg11
from torchvision.transforms import transforms
from tqdm import tqdm

if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    test_dataset = CIFAR10('data', train=False, transform=transform, download=False)
    test_dataset = Subset(test_dataset, range(9000, 9200))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3)

    net = vgg11(pretrained=False, num_classes=10).eval().cuda()
    net.load_state_dict(torch.load('saves/vgg11_cifar10.sv'))

    max_queries = 100
    n_tries = 10
    n_images = 1000

    f = open("results/random_v2_100_vgg.csv", 'w', newline='')
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

        pixels = Queue()

        total_queries = 0
        for t in range(n_tries):
            mask = torch.zeros_like(inputs)
            while n_queries < max_queries:
                x = random.randrange(0, 32)
                y = random.randrange(0, 32)

                if mask[:, :, x, y].mean() == 0:
                    pixels.put((x, y))
                    if pixels.qsize() > 50:
                        rx, ry = pixels.get()
                        mask[:, :, rx, ry] = 0

                    mask[:, :, x, y] = 1

                    out = net((inputs + mask).clamp(-1, 1)).argmax(dim=1)
                    if out != labels:
                        break

                    n_queries += 1
            if n_queries < max_queries - 1:
                n_correct += 1
            total_queries += n_queries

        csv_w.writerow((n_correct, total_queries))
        f.flush()

    f.close()