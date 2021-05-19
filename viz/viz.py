import plotly.graph_objs as go
from sklearn.decomposition import PCA
import numpy as np


def plot_trajectory(self, criterion, loader, xnum, ynum, use_visdom):
    """
    :param criterion: Loss criterion
    :param loader: Loader for tests
    :param xnum: Number of ticks on the x axis
    :param ynum: Number of ticks on the y axis
    :param use_visdom: If True then visdom will be used to display a plot
    :return: x: A xnum x ynum matrix with the x coordinate of each point
             y: A xnum x ynum matrix with the y coordinate of each point
             z: A xnum x ynum matrix with the loss value in each point
             reduced_x: A vector with the x values of the points in the trajectory
             reduced_y: A vector with the y values of the points in the trajectory
             lrChangePoints (optional): A vector with the indices of the trajectory where the learning rate changes
    """
    self.weights = self.trajectory[-1]
    translated_trajectory = self.trajectory - self.weights
    pca = PCA(n_components=2)
    pca.fit(translated_trajectory.detach().cpu().numpy())
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    pca.mean_ = None
    reduced_trajectory = pca.transform(translated_trajectory.detach().cpu().numpy())
    reduced_x = reduced_trajectory[:, 0]
    reduced_y = reduced_trajectory[:, 1]
    min_x, max_x, min_y, max_y = self._get_contour_coordinates(reduced_x, reduced_y)
    self.dir1 = torch.Tensor(pc1).to(self.device)
    self.dir2 = torch.Tensor(pc2).to(self.device)
    x, y, z = self.plot2D(criterion, loader, min_x, max_x, xnum, min_y, max_y, ynum, use_visdom=False)

    if self.visdom is not None and use_visdom:
        contour = go.Contour(x=x[0, :], y=y[:, 0], z=self._clip_losses(z), colorscale='Viridis',
                             contours=dict(coloring='fill'))
        line = go.Scatter(x=reduced_x, y=reduced_y, showlegend=False)
        endPoint = go.Scatter(x=[0], y=[0], marker=dict(color='red', size=12), mode='markers', showlegend=False)
        lrChanges = go.Scatter(x=reduced_x[self.lrChangePoints], y=reduced_y[self.lrChangePoints],
                               marker=dict(color='blue', size=10), mode='markers', showlegend=False)
        # layout parameter doesn't work on visdom version 0.1.8.8, works on 0.1.8.5
        # figure = go.Figure(data=[contour, line, lrChanges, endPoint], layout=dict(title='Trajectory'))
        figure = go.Figure(data=[contour, line, lrChanges, endPoint])
        self.visdom.plotlyplot(figure)
        surface = go.Surface(x=x[0, :], y=y[:, 0], z=self._clip_losses(z), colorscale='Viridis')
        figure = go.Figure(data=[surface], layout=dict(title='Surface'))
        self.visdom.plotlyplot(figure)

    return x, y, z, reduced_x, reduced_y, self.lrChangePoints
