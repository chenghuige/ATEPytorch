import os

import configs as gc

from torchtext import data
from torchtext.vocab import GloVe
from torchtext.data import Dataset
from torchtext.data.example import Example

class OnlineDataset(Dataset):

    def __init__(self, data, fields, **kwargs):
        """
        data: [[field_1, field_2...]...]
        """

        make_example = Example.fromCSV

        examples = [make_example(line, fields) for line in data]

        super(OnlineDataset, self).__init__(examples, fields, **kwargs)


class AEDataLoader:

    def __init__(self, tokenizer, glove="twitter.27B", embed_dim=200, preprocessor=None):

        self.glove = glove #  42B, 840B, twitter.27B, 6B
        self.embed_dim = embed_dim

        self.resource = os.path.join(gc.PROJECT_ROOT, "resources")
        self.vectors_path = os.path.join(self.resource, "vector_cache")

        if tokenizer is not None:
            self.TEXT = data.Field(sequential=True,
                                   batch_first=True,
                                   include_lengths=True,
                                   tokenize=tokenizer,
                                   lower=True,
                                   preprocessing=preprocessor)
        else:
            self.TEXT = data.Field(sequential=True,
                                   batch_first=True,
                                   include_lengths=True,
                                   lower=True,
                                   preprocessing=preprocessor)

        self.LABEL = data.Field(sequential=False,
                                lower=True,
                                batch_first=True)

# Static Aspect Extraction Data Loader
class StaticAEDataLoader(AEDataLoader):

    def __init__(self, dataname_type, glove="twitter.27B", embed_dim=25, tokenizer=None, preprocessor=None):

        # dataset: "dataset_train"
        super(StaticAEDataLoader, self).__init__(glove=glove,
                                                 embed_dim=embed_dim,
                                                 tokenizer=tokenizer,
                                                 preprocessor=preprocessor)

        self.data_path = {
            "semeval14laptop": {
                "train": "semeval14/laptops_train.csv",
                "test": "semeval14/laptops_test.csv"
            },
            "semeval14restaurant": {
                "train": "semeval14/restaurants_train.csv",
                "test": "semeval14/restaurants_test.csv"
            },
            "aedebug": {
                "train": "citysearch/debug_train.csv",
                "test": "citysearch/debug_test.csv"
            }, # small dataset for debugging
            "citysearch": {
                "train": "citysearch/train.csv",
                "test": "citysearch/test.csv"
            }, # citysearch data from ABAE paper with pre-processing
            "citysearchfull": {
                "train": "citysearch_full/train.csv",
                "test": "citysearch_full/test.csv"
            }, # original citysearch data
            "beeradvocate": {
                "train": "beeradvocate/train.csv",
                "test": "beeradvocate/test.csv"
            }
        }

        self.dataname, self.datatype = dataname_type.split("_")
        self.data_path = os.path.join(self.resource, self.data_path[self.dataname][self.datatype])

        self._data_reader()

        self.desc = ">>> Data Summary \n" \
               "\t-dataset: {}\n" \
               "\t-embedding dimension: {}\n" \
               "\t-train: {}\n" \
               "\t-distinct word: {}\n".format(dataname_type,
                                               embed_dim,
                                               self.data_size,
                                               self.word_size)
        print(self.desc)

    def _data_reader(self):

        if self.datatype == "test":
            dataset = data.TabularDataset(
                path=self.data_path, format='csv',
                fields=[('label', self.LABEL),
                        ('text', self.TEXT)])
            self.LABEL.build_vocab(dataset)
            self.itol = self.LABEL.vocab.itos
        elif self.datatype == "train":
            dataset = data.TabularDataset(
                path=self.data_path, format='csv',
                fields=[('text', self.TEXT)])
        else:
            raise Exception("datatype other than train or test...")

        self.TEXT.build_vocab(dataset, vectors=GloVe(name=self.glove, dim=self.embed_dim, cache=self.vectors_path))


        self.dataset = dataset
        self.data_size = len(dataset)
        self.embed_matrix = self.TEXT.vocab.vectors
        self.word_size = len(self.embed_matrix)
        self.stoi = self.TEXT.vocab.stoi
        self.itos = self.TEXT.vocab.itos


# Online Aspect Extraction Data Loader
class OnlineAEDataLoader(AEDataLoader):

    def __init__(self, data, label=False, glove="twitter.27B", embed_dim=25, tokenizer=None, preprocessor=None):

        # data: [[field_1, field_2...]...]
        self.data = data
        self.label=label
        super(OnlineAEDataLoader, self).__init__(glove=glove,
                                                 embed_dim=embed_dim,
                                                 tokenizer=tokenizer,
                                                 preprocessor=preprocessor)

        self._data_reader()

        self.desc = ">>> Data Summary \n" \
                    "\t-dataset: {}\n" \
                    "\t-embedding dimension: {}\n" \
                    "\t-train: {}\n" \
                    "\t-distinct word: {}\n".format("Online Data",
                                                    embed_dim,
                                                    self.data_size,
                                                    self.word_size)

    def _data_reader(self):

        if self.label:
            dataset = OnlineDataset(data=self.data,
                                    fields=[('label', self.LABEL),
                                            ('text', self.TEXT)])
            self.itol = self.LABEL.vocab.itos
        else:
            dataset = OnlineDataset(data=self.data,
                                    fields=[('text', self.TEXT)])

        self.TEXT.build_vocab(dataset, vectors=GloVe(name=self.glove, dim=self.embed_dim, cache=self.vectors_path))


        self.dataset = dataset
        self.data_size = len(dataset)
        self.embed_matrix = self.TEXT.vocab.vectors
        self.word_size = len(self.embed_matrix)
        self.stoi = self.TEXT.vocab.stoi
        self.itos = self.TEXT.vocab.itos

class Vocab:

    def __init__(self, glove="twitter.27B", embed_dim=200, tokenizer=None, preprocessor=None):
        self.glove = glove  # 42B, 840B, twitter.27B, 6B
        self.embed_dim = embed_dim

        self.vectors_path = os.path.join(os.path.dirname(gc.PROJECT_ROOT), "resources/vector_cache")


    def run(self, data):

        # data: [[field_1, field_2...]...]

        VOCAB = data.Field(batch_first=True,
                           lower=True)
        dataset = OnlineDataset(data=data,
                                fields=[('text', VOCAB)])
        VOCAB.build_vocab(dataset, vectors=GloVe(name=self.glove, dim=self.embed_dim, cache=self.vectors_path))

        return VOCAB.vocab.stoi, VOCAB.vocab.itos, VOCAB.vocab.vectors
