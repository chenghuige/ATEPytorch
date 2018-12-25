import os
import csv
import torch
import pickle
import codecs

import numpy as np
import configs as gc
import torch.nn as nn

from torch.optim import Adam
from layers.attention import Attention


class ABAE(nn.Module):

    def __init__(self, args, asp_mat):

        super(ABAE, self).__init__()
        self.args = args
        self.device = args.mdevice
        self.asp_mat = nn.Parameter(asp_mat, requires_grad=True) # [n_aspect, embed_dim]

        self.text_linear = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.attention = Attention(m_dim=args.embed_dim,
                                   score_func="sdp")
        self.asp_infer = nn.Sequential(
            nn.Linear(args.embed_dim, args.n_aspect),
            nn.Softmax(dim=-1)
        )

        self.desc = ">>> Model Summary [{}] \n" \
                    "\t-aspect: {}\n".format(self.__class__.__name__,
                                             args.n_aspect)

    def _avg(self, text, text_length):

        # text: [batch_size, doc_size, embed_dim]
        # text_length: [batch_size]

        sum = torch.sum(text, dim=1) # [batch_size, embed_dim]
        denominator = torch.max(torch.tensor(1, dtype=torch.float).to(self.device), text_length.float())

        return sum.div(denominator.unsqueeze(dim=1)) # [batch_size, embed_dim]

    def forward(self, *input):

        # text: [batch_size, doc_size, embed_dim]
        # text_length: [batch_size]
        text, text_length = input

        avg_text = self._avg(text, text_length) # [batch_size, embed_dim]
        key = self.text_linear(text) # [batch_size, doc_size, embed_dim]

        # attn_score: [batch_size, doc_size]
        # doc_rps: document representation [batch_size, embed_dim]
        attn_score, doc_reprs = self.attention(avg_text, key, text)

        asp_score = self.asp_infer(doc_reprs) # [batch_size, n_aspect]
        asp_recon = torch.matmul(asp_score, self.asp_mat) # [batch_size, embed_dim]

        outputs = dict()
        outputs["attn_score"] = attn_score
        outputs["zs"] = doc_reprs
        outputs["rs"] = asp_recon
        outputs["asp_mat"] = self.asp_mat

        return outputs

class ABAECriterion(nn.Module):

    def __init__(self, args):

        super(ABAECriterion, self).__init__()
        self.args = args
        self.ld = args.ld # lambda
        self.device = args.mdevice

    def __norm(self, vec):

        # len(vec.shape) = 2
        denorm = torch.norm(vec, p=2, dim=-1)
        denorm = torch.max(torch.tensor(1e-6, dtype=torch.float).to(self.device), denorm)

        return vec.div(denorm.unsqueeze(dim=1))

    def forward(self, *input):

        # neg_sample: [neg_size, embed_dim], average pooling on negative sample
        # zs: [batch_size, embed_dim]
        # rs: [batch_size, embed_dim]
        outputs, neg_sample = input
        zs = outputs["zs"]
        rs = outputs["rs"]
        t = outputs["asp_mat"]

        batch_size, _ = rs.shape
        neg_size, _ = neg_sample.shape

        # normalization
        neg_sample = self.__norm(vec=neg_sample)  # [neg_size, embed_dim]
        rs = self.__norm(vec=rs)  # [batch_size, embed_dim]
        zs = self.__norm(vec=zs)  # [batch_size, embed_dim]
        t = self.__norm(vec=t)  # [n_aspect, embed_dim]

        # cohesion(rs, zs)
        cohension = torch.sum(torch.mul(rs, zs), dim=1)  # [batch_size]
        # coupling(zs, neg_sample) or coupling(rs, neg_sample)
        neg_sample = neg_sample.repeat(batch_size, 1).view(batch_size, neg_size, -1)  # [batch_size, neg_size, embed_dim]
        # coupling = torch.sum(torch.matmul(zs.unsqueeze(dim=1), neg_sample.permute(0, 2, 1)).squeeze(dim=1), dim=1).div(neg_size) # [batch_size]
        # [batch_size] = [batch_size, 1, neg_size]
        coupling = torch.sum(torch.matmul(rs.unsqueeze(dim=1), neg_sample.permute(0, 2, 1)).squeeze(dim=1), dim=1).div(neg_size)  # [batch_size]
        # sum(max(0, 1-sim+neg))/batch_size
        loss = torch.sum(torch.max(torch.tensor(0, dtype=torch.float).to(self.device),torch.tensor(1, dtype=torch.float).to(self.device) - cohension + coupling)).div(batch_size)

        reg = torch.norm(torch.matmul(t, t.transpose(0, 1)) - torch.eye(self.args.n_aspect).to(self.device), p=2)

        # objective
        loss = loss + self.ld * reg

        return loss


class Trainer:

    def run(self):
        # init
        self.best_loss = None

        global_step = 0
        for i_epoch in range(self.epoch):
            train_iterator = iter(self.train_iter)
            sample_iterator = iter(self.sample_iter)

            while True:
                try:
                    # train process
                    global_step += 1

                    train_batch = next(train_iterator)
                    sample = next(sample_iterator)

                    train_loss = self._train(train_batch.text[0],train_batch.text[1], sample.text[0], sample.text[1])
                    if (self.best_loss is None) or (train_loss < self.best_loss):

                        self.best_loss = train_loss.data
                        print(">>> obtain lowest loss: {}, saving model...".format(self.best_loss))
                        # save model
                        torch.save(self.model, self.model_path)
                        torch.save(self.model.asp_mat.data, self.aspect_mat_path)
                        # save top_n infer matrix in the last epoch
                        self._aspect_output(aspect_matrix=self.model.asp_mat.data, save=True)

                    if global_step % self.log_step == 0:
                        desc = "step {}: {}/{}\n".format(global_step, self.best_loss, train_loss)
                        print(desc)
                        with codecs.open(self.log_path, "a+", "utf-8") as logfile:
                            logfile.write(desc)

                except StopIteration:
                    desc = ">>> epoch {} completed... \n".format(i_epoch)
                    print(desc)
                    with codecs.open(self.log_path, "a+", "utf-8") as logfile:
                        logfile.write(desc)
                    break

            with torch.no_grad():
                # aspect words output
                aspect_mat = self.model.asp_mat.data
                self._aspect_output(aspect_matrix=aspect_mat, save=False)

    def _avg_embed(self, text, text_length):

        # text: [*, doc_size, embed_dim]
        # text_length: [*]
        text_length = torch.max(torch.tensor(1, dtype=torch.float).to(self.device), text_length.float())
        return torch.sum(text, dim=1).div(text_length.unsqueeze(dim=1)) # [*, embed_dim]

    def _train(self, *inputs):

        self.model.train()
        self.optimizer.zero_grad()

        text, text_length, sample, sample_length = inputs
        text = self.train_embedding(text)
        sample = self.train_embedding(sample) # [neg_size, doc_size, embed_dim]
        sample_avg = self._avg_embed(sample, sample_length) # [neg_size, embed_dim]

        outputs = self.model(text, text_length)

        # criterion input: outputs
        loss = self.criterion(outputs, sample_avg)

        loss.backward()
        self.optimizer.step()

        return loss

    def _aspect_output(self, aspect_matrix, save=False):
        # aspect_matrix: [n_aspect, embed_dim]
        # normed aspect matrix
        aspect_mat = aspect_matrix.div(torch.max(torch.tensor(1e-6, dtype=torch.float).to(self.device), torch.norm(aspect_matrix, p=2, dim=-1)).unsqueeze(dim=1))

        output_desc = ">>> aspect words (top {})\n".format(self.top_n)
        infer_mat = torch.matmul(aspect_mat, self.normed_embedding_matrix.transpose(0, 1)).cpu().numpy() # [n_aspect, vocab_size]
        # infer_mat = torch.matmul(aspect_mat, self.embedding_matrix.transpose(0, 1)).div(self.scale_term).cpu().numpy()  # [n_aspect, vocab_size]
        topn_mat = [[self.itos[wi] for wi in row] for row in np.argsort(-infer_mat, 1)[:,:self.top_n]] # sorted, word mapped, [n_aspect, top_n]
        aspect_count = 0
        for aspect_infer in topn_mat:
            # [top_n]
            aspect_count += 1
            output_desc += "\t-aspect{}: ".format(aspect_count)
            for w in aspect_infer:
                output_desc += "{} | ".format(w)
            output_desc += "\n"

        print(output_desc)
        with codecs.open(self.log_path, "a+", "utf-8") as logfile:
            logfile.write(output_desc)

        if save:
            print(">>> saving topn aspect infering matrix to {}...".format(self.aspect_word_path))
            pickle.dump(topn_mat, open(self.aspect_word_path, 'wb'))
            with codecs.open(self.aspect_word_text_path, "w", "utf-8") as awf:
                awf.write(output_desc)

    def _init_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def __init__(self, args, embedding_matrix, itos, train_iter, sample_iter, test_iter, model, criterion):

        self.args = args
        self.device = args.mdevice

        # data iterator
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.test_iter = test_iter
        # model
        self.model = model.to(self.device)
        # criterion
        self.criterion = criterion
        # optimizer
        self.learning_rate = args.learning_rate
        params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = Adam(params, lr=args.learning_rate)

        # other params
        self.epoch = args.epoch
        self.log_step = args.log_step
        self.itos = itos
        self.normed_embedding_matrix = embedding_matrix.div(torch.max(torch.tensor(1e-6, dtype=torch.float).to(self.device), torch.norm(embedding_matrix, p=2, dim=-1)).unsqueeze(dim=1))
        self.scale_term = np.power(args.embed_dim, 0.5)
        self.top_n = args.top_n
        self.train_embedding = nn.Embedding.from_pretrained(embedding_matrix).to(self.device)

        # paths
        self._output_dir = os.path.join(gc.PROJECT_ROOT, "outputs")
        self.model_path = os.path.join(self._output_dir, "ABAE_{}_{}_model.pt".format(args.dataset, args.n_aspect))
        self.aspect_mat_path = os.path.join(self._output_dir,"ABAE_{}_{}_apmat.pt".format(args.dataset, args.n_aspect))
        self.aspect_word_path = os.path.join(self._output_dir, "ABAE_{}_top{}_apw.dat".format(args.dataset, args.top_n))
        self.aspect_word_text_path = os.path.join(self._output_dir, "ABAE_{}_top{}_apw.txt".format(args.dataset, args.top_n))

        self._log_dir = os.path.join(gc.PROJECT_ROOT, "log")
        self.log_path = os.path.join(self._log_dir, "ABAE_{}_{}.txt".format(args.dataset, args.n_aspect))

        self._init_dir(dir_path=self._output_dir)
        self._init_dir(dir_path=self._log_dir)

        # remove log file as initialization
        if os.path.exists(self.log_path):
            os.remove(self.log_path)