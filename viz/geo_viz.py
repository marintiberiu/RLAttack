import geomstats.backend as gs
import torch
from geomstats import visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.preprocessing import ToTangentSpace
from geomstats.visualization import Sphere
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == '__main__':
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5)))
    test_dataset = CIFAR10('data', train=True, transform=transform, download=False)
    net = resnet18(pretrained=False, num_classes=10).eval()
    net.load_state_dict(torch.load('saves/resnet18_cifar10.sv'))

    image_size = 32 * 32 * 3
    num_images = 3000

    features = np.zeros((0, image_size))
    labels = []
    for idx, data in enumerate(tqdm(test_dataset)):
        if idx == num_images:
            break
        features = np.concatenate((features, data[0].reshape((1, image_size)).numpy()))
        labels.append(data[1])

    labels = np.array(labels)
    pca = PCA(n_components=2)
    features = pca.fit_transform(features)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)

    scaler = StandardScaler()
    features = scaler.fit_transform(features) * 3
    features = np.concatenate((features, np.zeros_like(features[:, 0:1])), axis=1)

    sphere = Hypersphere(dim=2)
    transformer = ToTangentSpace(sphere)
    base_point = np.array([0, 0, 1])
    points = transformer.inverse_transform(features, base_point)
    entropy_loss = CrossEntropyLoss()

    def loss_f(x):
        x = x[np.newaxis, :]
        x = transformer.inverse_transform(x, base_point)
        x = scaler.inverse_transform(x[:, :2] / 3)
        x = pca.inverse_transform(x)
        x = torch.from_numpy(x).reshape((1, 3, 32, 32)).float()
        with torch.no_grad():
            x = net(x)
        return entropy_loss(x, torch.zeros(1, dtype=torch.long))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    sphere = Sphere()

    # f_points = sphere.fibonnaci_points(10000).swapaxes(0, 1)
    # tf_points = transformer.transform(f_points, base_point)
    #
    # print(tf_points[:, 0].min(), tf_points[:, 0].max())
    # print(tf_points[:, 1].min(), tf_points[:, 1].max())
    # print(tf_points[:, 2].min(), tf_points[:, 2].max())

    sphere.plot_heatmap(ax=ax, n_points=10000, scalar_function=loss_f)
    correct_points = points[labels == 0]
    correct_labels = labels[labels == 0]
    wrong_points = points[labels != 0]
    wrong_labels = labels[labels != 0]
    plot_points = np.concatenate((correct_points[:100], wrong_points[:100]))
    plot_labels = np.concatenate((correct_labels[:100], wrong_labels[:100]))
    visualization.plot(plot_points, ax=ax, space='S2', c=plot_labels == 0, s=80, alpha=0.5)
    plt.show()
