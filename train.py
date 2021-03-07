import sys
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from ETIVDataSet import ETIVDataSet
# from torch.nn.functional import softmax
from torch.utils.data import DataLoader

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0

class ETIV:
    def __init__(self, file_path, emb_size=128, neg_size=10, hist_len=5, directed=False):

        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = 0.001
        self.batch = 128
        self.save_step = 5
        self.epochs = 50

        self.data = ETIVDataSet(file_path, neg_size, hist_len, directed)
        self.node_dim = self.data.get_node_dim()

        self.alpha = 1.
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if torch.cuda.is_available():
            with torch.cuda.device(DID):

                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)

                self.W_1 = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)).type(FType).cuda(),
                                        requires_grad=True)
                nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
                self.W_2 = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)).type(FType).cuda(),
                                        requires_grad=True)
                nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
                self.W_3 = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)).type(FType).cuda(),
                                        requires_grad=True)
                nn.init.xavier_uniform_(self.W_3.data, gain=1.414)

                self.a = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).type(FType).cuda(), requires_grad=True)
                nn.init.xavier_uniform_(self.a.data, gain=1.414)
                self.b = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).type(FType).cuda(), requires_grad=True)
                nn.init.xavier_uniform_(self.b.data, gain=1.414)
                self.c = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).type(FType).cuda(), requires_grad=True)
                nn.init.xavier_uniform_(self.c.data, gain=1.414)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

        else:

            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.W_1 = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)).type(FType), requires_grad=True)
            nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
            self.W_2 = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)).type(FType), requires_grad=True)
            nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
            self.W_3 = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)).type(FType), requires_grad=True)
            nn.init.xavier_uniform_(self.W_3.data, gain=1.414)
            self.a = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).type(FType), requires_grad=True)
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
            self.b = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).type(FType), requires_grad=True)
            nn.init.xavier_uniform_(self.b.data, gain=1.414)
            self.c = nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).type(FType), requires_grad=True)
            nn.init.xavier_uniform_(self.c.data, gain=1.414)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

        self.opt = Adam(lr=self.lr, params=[self.delta, self.node_emb,
                                            self.W_1, self.W_2, self.W_3,
                                            self.a, self.b, self.c])
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):

        batch = s_nodes.size()[0]

        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)


        s_node_emb_h_1 = torch.mm(s_node_emb, self.W_1).unsqueeze(1)
        s_node_emb_h_1 = s_node_emb_h_1.repeat(1, self.hist_len, 1)
        h_node_emb_h_1 = torch.matmul(h_node_emb, self.W_1)
        contact_s_h_1 = torch.cat([s_node_emb_h_1, h_node_emb_h_1], dim=-1)
        mid_e_1 = torch.matmul(contact_s_h_1, self.a).squeeze(2)
        e_1 = torch.tanh(mid_e_1)
        e_1 = torch.mul(torch.exp(e_1), h_time_mask)
        att_1 = torch.div(e_1, (e_1.sum(dim=1) + 1e-6).unsqueeze(1))  # (batch, hist_len)

        s_node_emb_h_2 = torch.mm(s_node_emb, self.W_2).unsqueeze(1)
        s_node_emb_h_2 = s_node_emb_h_2.repeat(1, self.hist_len, 1)
        h_node_emb_h_2 = torch.matmul(h_node_emb, self.W_2)
        contact_s_h_2 = torch.cat([s_node_emb_h_2, h_node_emb_h_2], dim=-1)
        mid_e_2 = torch.matmul(contact_s_h_2, self.a).squeeze(2)
        e_2 = torch.tanh(mid_e_2)
        e_2 = torch.mul(torch.exp(e_2), h_time_mask)
        att_2 = torch.div(e_2, (e_2.sum(dim=1) + 1e-6).unsqueeze(1))

        s_node_emb_h_3 = torch.mm(s_node_emb, self.W_3).unsqueeze(1)
        s_node_emb_h_3 = s_node_emb_h_3.repeat(1, self.hist_len, 1)
        h_node_emb_h_3 = torch.matmul(h_node_emb, self.W_3)
        contact_s_h_3 = torch.cat([s_node_emb_h_3, h_node_emb_h_3], dim=-1)
        mid_e_3 = torch.matmul(contact_s_h_3, self.a).squeeze(2)
        e_3 = torch.tanh(mid_e_3)
        e_3 = torch.mul(torch.exp(e_3), h_time_mask)
        att_3 = torch.div(e_3, (e_3.sum(dim=1) + 1e-6).unsqueeze(1))

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)

        Time_decay_coefficient = (torch.exp((delta * Variable(d_time)).neg()) * Variable(h_time_mask)).unsqueeze(2)
        h_node_emb_T_d_c = torch.mul(Time_decay_coefficient, h_node_emb)  # 节点表征随时间衰减


        h_node_emb_mid_f_1 = torch.matmul(h_node_emb_T_d_c, self.W_1)
        h_node_emb_f_1 = ((torch.mul(att_1.unsqueeze(2), h_node_emb_mid_f_1)).sum(dim=1)).sigmoid()

        h_node_emb_mid_f_2 = torch.matmul(h_node_emb_T_d_c, self.W_2)
        h_node_emb_f_2 = ((torch.mul(att_2.unsqueeze(2), h_node_emb_mid_f_2)).sum(dim=1)).sigmoid()

        h_node_emb_mid_f_3 = torch.matmul(h_node_emb_T_d_c, self.W_3)
        h_node_emb_f_3 = ((torch.mul(att_3.unsqueeze(2), h_node_emb_mid_f_3)).sum(dim=1)).sigmoid()

        h_node_emb_f = (h_node_emb_f_1 + h_node_emb_f_2 + h_node_emb_f_3)/3.0  # (batch, -1)

        s_new_emb = h_node_emb_f

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_lambda = p_mu + ((s_new_emb - t_node_emb) ** 2).sum(dim=1).neg()
        n_lambda = n_mu + ((s_new_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()

        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)

                loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)

        else:
            p_lambdas, n_lambdas= self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times,
                                                h_time_mask)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()

        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):

            once_start = datetime.datetime.now()
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=4)

            if epoch % self.save_step == 0 and epoch != 0:
                # torch.save(self, './model/dnrl-dblp-%d.bin' % epoch)
                self.save_node_embeddings('./emb/dblp_histlen5/dblp_etiv_attn_%d.emb' % (epoch))

            for i_batch, sample_batched in enumerate(loader):

                if i_batch % 100 == 0 and i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))

                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))
            once_end = datetime.datetime.now()

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) +
                             '\tonce_runtime: ' + str(once_end - once_start) + '\n')
            sys.stdout.flush()

        self.save_node_embeddings('./emb/dblp_histlen5/dblp_etiv_attn_%d.emb' % (self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()

        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))

        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()


if __name__ == '__main__':
    etiv = ETIV('./data/dblp/dblp.txt', directed=False)
    etiv.train()
