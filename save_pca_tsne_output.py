import os
from urllib import request
import tarfile
import scipy.io
import keras_flower as kf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

DATA_DIR = 'data'
FLOWER_DIR = os.path.join(DATA_DIR, 'flowers')
FLOWER_TGZ = FLOWER_DIR + '.tgz'
LABELS_FILE = os.path.join(DATA_DIR, 'imagelabels.mat')


def create_dir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)


def download_data_if_not_exists():
    if not os.path.exists(FLOWER_DIR):
        if not os.path.exists(FLOWER_TGZ):
            _url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
            request.urlretrieve(_url, filename=FLOWER_TGZ)
            request.urlretrieve('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', LABELS_FILE)
        tar = tarfile.open(FLOWER_TGZ, "r:gz")
        create_dir_if_not_exists(FLOWER_DIR)
        tar.extractall(FLOWER_DIR)
        tar.close()


def get_labels():
    return scipy.io.loadmat(LABELS_FILE)['labels'][0]


def get_images():
    return [os.path.join(FLOWER_DIR, 'jpg', x) for x in os.listdir(os.path.join(FLOWER_DIR, 'jpg'))]


def get_embeds():
    _file = 'embeds.npy'
    if os.path.exists(_file):
        embeds = np.load(file=_file, allow_pickle=True)
    else:
        download_data_if_not_exists()
        embeds = [kf.embed_by_path(_x) for _x in get_images()]
        np.save(_file, embeds, allow_pickle=True)
    return embeds


def get_pca():
    _file = 'pca_embeds.npy'
    if os.path.exists(_file):
        pca_values = np.load(_file, allow_pickle=True)
    else:
        _embeds = get_embeds()
        pca_values = PCA(n_components=3).fit_transform(_embeds)
        np.save(_file, pca_values, allow_pickle=True)
    return pca_values


def get_tsne():
    _file = 'tsne_embeds.npy'
    if os.path.exists(_file):
        tsne_values = np.load(_file, allow_pickle=True)
    else:
        _embeds = get_embeds()
        tsne_values = TSNE(n_components=3).fit_transform(_embeds)
        np.save(_file, tsne_values, allow_pickle=True)
    return tsne_values


if __name__ == '__main__':
    create_dir_if_not_exists(DATA_DIR)
    pca_df = pd.DataFrame(data=get_pca())
    print(pca_df)

    tsne_df = pd.DataFrame(data=get_tsne())
    print(tsne_df)
    pass
