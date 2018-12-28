import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CMLAUnit(nn.Module):

    def __init__(self, K=20, hidden_dim=50):

        super(CMLAUnit, self).__init__()

        self.K = K
        self.hidden_dim = hidden_dim

        self.G_a = nn.Linear(hidden_dim, K * hidden_dim)
        self.G_p = nn.Linear(hidden_dim, K * hidden_dim)
        self.D_a = nn.Linear(hidden_dim, K * hidden_dim)
        self.D_p = nn.Linear(hidden_dim, K * hidden_dim)

        self.gru_ra = nn.GRU(input_size=2*K,
                            hidden_size=2*K,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        self.gru_rp = nn.GRU(input_size=2 * K,
                             hidden_size=2 * K,
                             num_layers=1,
                             bias=True,
                             batch_first=True,
                             bidirectional=self.bidirectional)

    def forward(self, *input):

        (h, h_length), (u_a, u_p) = input
        # h: [batch_size, doc_size, hidden_dim]
        # u_a: [hidden_dim]
        # u_p: [hidden_dim]

        b_s, d_s, h_d = h.shape
        # [batch_size, doc_size, K] <= [batch_size, doc_size, K, hidden_dim] * [hidden_dim, 1]
        h_G_ua = torch.matmul(self.G_a(h).view(b_s, d_s, self.K, h_d), u_a.unsqueeze(1)).squeeze(3)
        h_D_up = torch.matmul(self.D_a(h).view(b_s, d_s, self.K, h_d), u_p.unsqueeze(1)).squeeze(3)
        h_G_up = torch.matmul(self.G_p(h).view(b_s, d_s, self.K, h_d), u_p.unsqueeze(1)).squeeze(3)
        h_D_ua = torch.matmul(self.D_p(h).view(b_s, d_s, self.K, h_d), u_a.unsqueeze(1)).squeeze(3)
        # [batch_size, doc_size, 2K]
        fa = F.tanh(torch.cat((h_G_ua, h_D_up), -1))
        fp = F.tanh(torch.cat((h_G_up, h_D_ua), -1))

        # GRU_fa
        fa_pack = pack_padded_sequence(fa, h_length, batch_first=True)
        grura_out, _ = self.gru_ra(fa_pack, None)
        ra, _ = pad_packed_sequence(grura_out, batch_first=True)
        ra = ra.view(b_s, d_s, -1) # [batch_size, doc_size, 2K]

        # GRU_fp
        fp_pack = pack_padded_sequence(fp, h_length, batch_first=True)
        grurp_out, _ = self.gru_rp(fp_pack, None)
        rp, _ = pad_packed_sequence(grurp_out, batch_first=True)
        rp = rp.view(b_s, d_s, -1) # [batch_size, doc_size, 2K]

        # scalar score
        ea = torch.matmul(ra, fa.unsqueeze(3)).squeeze(2) # [batch_size, doc_size]
        ep = torch.matmul(rp, fp.unsqueeze(3)).squeeze(2) # [batch_size, doc_size]

        return (ra, rp), (ea, ep)


class CMLA(nn.Module):

    def __init__(self, args):

        super(CMLA, self).__init__()
        self.args = args
        self.device = args.mdevice

        self.bidirectional = False
        self.num_bidirectional = 2 if self.bidirectional else 1
        self.hidden_dim = 50
        self.K = args.K

        self.gru_h = nn.GRU(input_size=args.embed_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        self.ua = nn.Parameter(torch.FloatTensor(self.hidden_dim), requires_grad=True)
        self.up = nn.Parameter(torch.FloatTensor(self.hidden_dim), requires_grad=True)

        self.cmla_layer1 = CMLAUnit()
        self.cmla_layer2 = CMLAUnit()

        self.Va = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Vp = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.aspect_proj = nn.Linear(2 * self.K, 3)
        self.opinion_proj = nn.Linear(2 * self.K, 3)

    def forward(self, *input):

        text, text_length = input
        # text: [batch_size, doc_size, embed_dim]
        # text_length: [batch_size]
        b_s, d_s, e_d = text.shape

        # GRU: [batch_size, doc_size, hidden_dim], [batch_size]
        pack_text = pack_padded_sequence(text, text_length, batch_first=True)
        gruh_out, _ = self.gru_h(pack_text, None)
        h, h_length = pad_packed_sequence(gruh_out, batch_first=True).view(b_s, d_s, -1) # (output, lengths)

        # first layer
        _, (ea, ep) = self.cmla_layer1((h, h_length), (self.ua, self.up))

        # aspect
        # usually, batch_size = 1
        oa = torch.matmul(F.softmax(ea, dim=-1), h).squeeze(0) # [hidden_dim] <= [1, hidden_dim]
        self.ua = F.tanh(self.Va(self.ua)) + oa

        # opinion
        # usually, batch_size = 1
        op = torch.matmul(F.softmax(ep, dim=-1), h).squeeze(0) # [hidden_dim] <= [1, hidden_dim]
        self.up = F.tanh(self.Vp(self.up)) + op

        # second layer
        (ra, rp), _ = self.cmla_layer2((h, h_length), (self.ua, self.up))
        la = F.softmax(self.aspect_proj(ra), -1) # [batch_size, doc_size, 3]
        lp = F.softmax(self.opinion_proj(rp), -1) # [batch_size, doc_size, 3]

        return la, lp