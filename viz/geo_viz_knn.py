import geomstats.backend as gs
import torch
from geomstats import visualization
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.learning.knn import KNearestNeighborsClassifier
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

    scaler = StandardScaler()
    features = scaler.fit_transform(features) * 3
    features = np.concatenate((features, np.zeros_like(features[:, 0:1])), axis=1)

    sphere = Hypersphere(dim=2)
    transformer = ToTangentSpace(sphere)
    base_point = np.array([0, 0, 1])
    s_points = transformer.inverse_transform(features, base_point)

    classifier = KNearestNeighborsClassifier()
    classifier.fit(s_points, labels)

    f_labels = []

    def loss_f(x):
        label = classifier.predict(x[np.newaxis, :])
        f_labels.append(label)
        return label == 0

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_sphere = Sphere()

    n_points = 10000
    f_points = plot_sphere.fibonnaci_points(n_points).swapaxes(0, 1)
    plot_sphere.plot_heatmap(ax=ax, n_points=n_points, scalar_function=loss_f)
    correct_points = s_points[labels == 0][:30, :]
    correct_labels = np.ones_like(correct_points)

    ax = visualization.plot(correct_points, ax=ax, space='S2', color='red', s=80)

    f_labels = np.array(f_labels)[:, 0]
    f_points = f_points[f_labels != 0]

    metric = HypersphereMetric(dim=2)
    for k in range(len(correct_points)):
        point_matrix = correct_points[k:k+1, :].repeat(len(f_points), axis=0)
        dist_array = metric.dist(point_matrix, f_points)
        idx_min = np.argmin(dist_array)

        geodesic = sphere.metric.geodesic(
            initial_point=correct_points[k],
            end_point=f_points[idx_min])

        points_on_geodesic = geodesic(gs.linspace(0., 1., 10))
        plot_sphere.add_points(points_on_geodesic)

    plot_sphere.draw_points(ax=ax, color='black', alpha=0.1)

    plt.show()
