import re
import codecs

from torchtext import data
from data_utils import StaticAEDataLoader
from models.ABAE import ABAE, ABAECriterion, Trainer
from utils import k_means_init, random_init
from nltk.tokenize import word_tokenize
from utils import debugparser

def tokenizer(string):

    string = re.sub(r'\d+', "", string)
    string = re.sub(r"[^A-Za-z0-9!?\']", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)

    return word_tokenize(string)

if __name__ == "__main__":

    args = debugparser(epoch=14)

    desc = "#### ABAE Model: An Unsupervised Neural Attention Model for Aspect Extraction ####\n"
    print(desc)

    trainloader = StaticAEDataLoader(dataname_type=args.dataset+"_train",
                                     glove=args.glove,
                                     embed_dim=args.embed_dim,
                                     tokenizer=tokenizer,
                                     preprocessor=None)

    testloader = StaticAEDataLoader(dataname_type=args.dataset + "_test",
                                    glove=args.glove,
                                    embed_dim=args.embed_dim,
                                    tokenizer=tokenizer,
                                    preprocessor=None)

    train_iter = data.Iterator(dataset=trainloader.dataset,
                               train=True,
                               batch_size=args.batch_size,
                               shuffle=True,
                               repeat=False,
                               device=args.ddevice)
    test_iter = data.Iterator(dataset=testloader.dataset,
                              train=False,
                              batch_size=len(testloader.dataset),
                              shuffle=False,
                              repeat=False,
                              sort=False,
                              sort_within_batch=False,
                              device=args.ddevice)
    sample_iter = data.Iterator(dataset=trainloader.dataset,
                                batch_size=args.neg_sample,
                                shuffle=True,
                                repeat=False,
                                device=args.ddevice)

    #### initialization ####
    aspect_mat = k_means_init(embedding_matrix=trainloader.embed_matrix,
                                   k=args.n_aspect)

    #### model ####
    model = ABAE(args=args,
                 asp_mat=aspect_mat)
    print(model.desc)

    #### train ####
    criterion = ABAECriterion(args=args)
    trainer = Trainer(args=args,
                      embedding_matrix=trainloader.embed_matrix.to(args.ddevice),
                      itos=trainloader.itos,
                      train_iter=train_iter,
                      sample_iter = sample_iter,
                      test_iter=test_iter,
                      model=model,
                      criterion=criterion)

    trainer.run()

    # write model info
    with codecs.open(trainer.log_path, "a+", "utf-8") as logfile:
        logfile.write(trainloader.desc)
        logfile.write(model.desc)