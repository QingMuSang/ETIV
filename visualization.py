from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import lil_matrix
import json
from time import time
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn import decomposition
import pandas as pd


def format_training_data_for_dnrl(emb_file, i2l_file):
    i2l = dict()
    with open(i2l_file, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            i2l[n_id] = l_id
    i2e = dict()
    with open(emb_file, 'r') as reader:
        reader.readline()
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_id = embeds[0]
            if node_id in i2l:
                i2e[node_id] = embeds[1:]

    a = 0
    b = 0
    c = 0
    d = len(i2l)
    i3l = []
    i3e = []
    for i in range(d):
        if i2l[i] == 4 and a < 500:
            i3l.append(i2l[i])
            i3e.append(i2e[i])
            a=a+1
        if i2l[i] == 7 and b < 500:
            i3l.append(i2l[i])
            i3e.append(i2e[i])
            b=b+1
        if i2l[i] == 5 and c < 500:
            i3l.append(i2l[i])
            i3e.append(i2e[i])
            c=c+1
        else:
            continue
    i3e =np.stack(i3e)

    return i3e, i3l


if __name__ == '__main__':
    X, Y = format_training_data_for_dnrl('./emb/dblp_htne/dblp_htne_attn_35.emb', './data/dblp/node2label.txt')
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', perplexity=30, random_state=0)
    reslut = ts.fit_transform(X)
    pos = pd.DataFrame(reslut, index=Y, columns=['X', 'Y'])

    ax=pos[pos.index == 4].plot(kind='scatter', x='X', y='Y', color='mediumseagreen', marker="^", s=6)
    pos[pos.index == 7].plot(kind='scatter', x='X', y='Y', color='royalblue', marker="^", s=6, ax=ax)
    pos[pos.index == 5].plot(kind='scatter', x='X', y='Y', color='mediumorchid', marker="^", s=6, ax=ax)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
