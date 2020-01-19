import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()

def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    n = X.shape[0]
    dist_matrix = euclidean_distances(X, X, squared=True)
    center_mat = np.identity(n) - (1/n * np.ones([n, n]))
    S = -0.5 * (center_mat @ dist_matrix @ center_mat)
    eig_vals, eig_vecs = np.linalg.eig(S)
    eig_vals_sorted = np.sort(eig_vals)
    eig_vecs_sorted = eig_vecs[:, eig_vals.argsort()]

    reduced_mat = np.zeros([n, d])
    for i in range(d):
        reduced_mat[:, i] = eig_vecs_sorted[i] * (eig_vals_sorted[i] ** 0.5)

    return reduced_mat

def generate_weight_matrix(X, k):
    N = X.shape[0]
    tree = KDTree(X)
    weight_matrix = np.zeros([N, N])
    for i in range(N):
        x_i = X[i]
        neighbors_idx = tree.query(x_i.reshape(1, -1), k=k + 1, return_distance=False)
        neighbors_idx = np.delete(neighbors_idx, [0]) # taking 1 more neighbour and ignoring x_i
        x_i_neighbors = X[neighbors_idx]
        z_vectors = x_i_neighbors - x_i
        grahm_mat = z_vectors @ z_vectors.T
        wights = np.linalg.pinv(grahm_mat)[0]  # solving
        wights /= np.sum(wights)  # normalizing
        weight_matrix[i, neighbors_idx] = wights  # zeros in all places but X_i's neighbors

    return weight_matrix

def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    W = generate_weight_matrix(X, k)
    M = np.identity(W.shape[0]) - W
    eig_vals, eig_vecs = np.linalg.eig(np.transpose(M) @ M)
    eig_vecs_indexes = eig_vals.argsort()[:-1]
    eig_vecs_indexes = eig_vecs_indexes[-d:]
    return eig_vecs[:, eig_vecs_indexes]




def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''

    n = X.shape[0]
    dist_matrix = euclidean_distances(X, X, squared=True)
    heat_kernel = np.exp(- dist_matrix / (2. * sigma ** 2))
    row_sums = heat_kernel.sum(axis=1)
    A = heat_kernel / row_sums[:, np.newaxis]
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals_sorted = np.sort(eig_vals)[1: d+1]
    eig_vecs_sorted = eig_vecs[:, eig_vals.argsort()]

    reduced_mat = np.zeros([n, d])
    for i in range(d):
        reduced_mat[:, i] = eig_vecs_sorted[i] * (eig_vals_sorted[i] ** t)

    return reduced_mat


if __name__ == '__main__':


    X, color = datasets._samples_generator.make_swiss_roll(n_samples=2000)

    for t in np.arange(0.01, 0.5, 0.05):
        for sig in np.arange(1, 10, 1):
            ans = DiffusionMap(X, 2, sig, t)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'DiffusionMap of swiss roll. sigma={sig}, t={t}')
            ax.scatter(ans[:, 0], ans[:, 1], c=color, cmap=plt.cm.Spectral)
            plt.show()

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()

    pass
