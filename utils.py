import os
import torch
import codecs
import argparse

import configs as gc

from sklearn.cluster import KMeans
from torch.nn.init import xavier_uniform_


# construct stop list
def stopword_list_constructor(stopwords=False):

    stopword_list = []
    # file path definition
    stopword_file = os.path.join(os.path.dirname(gc.PROJECT_ROOT), "resources/stopwords.txt")
    punctuation_file = os.path.join(os.path.dirname(gc.PROJECT_ROOT), "resources/punctuation.txt")

    if os.path.exists(punctuation_file):
        # if punctuation file exist
        with codecs.open(punctuation_file, "r", "utf-8") as file:
            for line in file.readlines():
                stopword_list.append(line.strip())
    else:
        raise Exception("punctuation file from default path ({}) not exists ...".format(punctuation_file))

    if stopwords:
        if os.path.exists(stopword_file):
            # if stopword file exist
            with codecs.open(stopword_file, "r", "utf-8") as file:
                for line in file.readlines():
                    stopword_list.append(line.strip())
        else:
            raise Exception("stopword file from default path ({}) not exists ...".format(stopword_file))

    return stopword_list

# preprocess, remove only punctuation
def pp(text):

    stopword_list = stopword_list_constructor(stopwords=False)

    return [word for word in text if word not in stopword_list]

# preprocess, remove punctuation and stopwords (including transitional words, which is needed by SA task)
def pp_word(text):

    stopword_list = stopword_list_constructor(stopwords=True)

    return [word for word in text if (word not in stopword_list and len(word)>2)]

def k_means_init(embedding_matrix, k):
    # input: embedding_matrix, number of cluster center(k)
    # output: cluster center matrix

    print(">>> Processing K-Means aspect embedding initialization...\n")

    estimator = KMeans(init="k-means++", n_clusters=k)
    estimator.fit(embedding_matrix)

    return torch.from_numpy(estimator.cluster_centers_).float()

def random_init(k, embed_dim):

    init = torch.empty(k, embed_dim)

    return xavier_uniform_(init)

def AVGEmbedding(text, text_length, device):

    # text: [*, doc_size, embed_dim]
    # text_length: [*]
    text_length = torch.max(torch.tensor(1, dtype=torch.float).to(device), text_length.float())
    return torch.sum(text, dim=1).div(text_length.unsqueeze(dim=1)) # [*, embed_dim]


def debugparser(dataset="citysearch", glove="twitter.27B", embed_dim=200, batch_size=64, log_step=1000, n_aspect=14,
             ld=1, epoch=50, learning_rate=0.001, neg_sample=20):

    parser = argparse.ArgumentParser(description='AE with Fine-Grained Attention')
    # data
    parser.add_argument("-dataset", type=str, default=dataset, help="dataset name [aedebug, citysearch]")
    parser.add_argument("-glove", type=str, default=glove,
                        help="GloVe, 42B, 840B, twitter.27B, 6B, [default: twitter.27B]")
    parser.add_argument("-embed_dim", type=int, default=embed_dim, help="number of embedding dimension")
    parser.add_argument("-batch_size", type=int, default=batch_size, help="batch size for training")

    # model
    parser.add_argument("-n_aspect", type=int, default=n_aspect, help="aspect cluster count")

    # train
    parser.add_argument("-neg_sample", type=float, default=neg_sample, help="weight of regularization term")
    parser.add_argument("-ld", type=float, default=ld, help="weight of regularization term")
    parser.add_argument("-epoch", type=int, default=epoch, help="train epoch")
    parser.add_argument("-log_step", type=int, default=log_step, help="log step [defalut: 256]")
    parser.add_argument("-learning_rate", type=float, default=learning_rate, help="learning rate")

    # output
    parser.add_argument("-top_n", type=int, default=20, help="top n words from aspect inference")
    args = parser.parse_args()
    args.ddevice = torch.device("cuda") if torch.cuda.is_available() else None
    args.mdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

