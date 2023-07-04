import torch
import numpy as np
import argparse
import os.path
from utils import prepare_data_gae
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import pdb
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GAE, VGAE, ARGVA
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-15

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        hidden_channels = 64
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels)#, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index), self.conv2(x, torch.tensor([[],[]],dtype=torch.int64).to(edge_index.device))

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        hidden_channels=64
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x

def recon_mul_loss(z, pos_edge_index, deg_wei, device, neg_edge_index = None):
    decoder = InnerProductDecoder()
    pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)).to(device)
    row, col = neg_edge_index
    neg_loss = -(deg_wei[row]*deg_wei[col]* torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS)).mean()

    return pos_loss + neg_loss, neg_edge_index

def compute_scores(z, test_pos, test_neg):
    test = torch.cat((test_pos, test_neg), dim=1)
    labels = torch.zeros(test.size(1), 1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt, dim=1))
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v.lower() == 'none':
        return None
    else:
        return str(v)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='HAW for GAEs')
# Dataset
parser.add_argument('--data-name', default='USAir', help='graph name')

# training/validation/test divison and ratio
parser.add_argument('--use-splitted', type=str2bool, default=True,
                    help='use the pre-splitted train/test data,\
                     if False, then make a random division')
parser.add_argument('--data-split-num', type=str, default='10',
                    help='If use-splitted is true, choose one of splitted data')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
# setups in peparing the training set
parser.add_argument('--observe-val-and-injection', type=str2bool, default=True,
                    help='whether to contain the validation set in the observed graph and apply injection trick')

parser.add_argument('--embedding-dim', type=int, default=32,
                    help='Dimension of the initial node representation, default: 32)')

# Model and Training
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--walk-len', type=int, default=7, help='cutoff in the length of walks')
parser.add_argument('--heads', type=int, default=2,
                    help='using multi-heads in the attention link weight encoder ')
parser.add_argument('--hidden-channels', type=int, default=32)
parser.add_argument('--epoch-num', type=int, default=500)
parser.add_argument('--MSE', type=str2bool, default=False)

parser.add_argument('--gpu-num', type=str, default='0',
                    help='Decide which gpu to train on')
parser.add_argument('--model', type=str, default='gae',
                    help='options: gae, vgae, argva')
parser.add_argument('--haw',type=str2bool, default=False,
                    help='whether to use HAW to enhance models ')
parser.add_argument('--non-uni',type=str2bool, default=False,
                    help='whether to use non-uniform sampling')

args = parser.parse_args()

device = torch.device('cuda:' + args.gpu_num if torch.cuda.is_available() else 'cpu')
print('Device:', device)

if args.data_name in ('cora', 'citeseer', 'pubmed', 'chameleon'):
    args.use_splitted = False
    args.observe_val_and_injection = False

#in non-uniform setting there are more low-degree nodes, enable o_v_a_i will decrease all the models' performance significantly
if args.non_uni:
    args.observe_val_and_injection = False

if (args.data_name in ('PB', 'pubmed')) and (args.max_nodes_per_hop == None):
    args.max_nodes_per_hop = 100

print("-" * 50 + 'Dataset and Features' + "-" * 60)
print("{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<20}" \
      .format('Dataset', 'Test Ratio', 'Val Ratio', 'Split Num', 'Dimension', \
              'Observe val and injection'))
print("-" * 130)
print("{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<20}" \
      .format(args.data_name, args.test_ratio, args.val_ratio, \
              args.data_split_num, args.embedding_dim,str(args.observe_val_and_injection)))
print("-" * 130)

print('<<Begin generating training data>>')

train_pos, edge_index, x, val_and_test, neg_edge_index, adj2, tneg_row, tneg_col = prepare_data_gae(args)

print('<<Complete generating training data>>')

print("-" * 42 + 'Model and Training' + "-" * 45)
print("{:<13}|{:<13}|{:<13}|{:<8}|{:<13}|{:<8}|{:<15}" \
      .format('Learning Rate', 'Weight Decay', 'Batch Size', 'Epoch', \
              'Walk Length', 'Heads', 'Hidden Channels'))
print("-" * 105)

print("{:<13}|{:<13}|{:<8}|{:<13}|{:<8}|{:<15}" \
      .format(args.lr, args.weight_decay, \
              args.epoch_num, args.walk_len, args.heads, args.hidden_channels))
print("-" * 105)

walk_len = args.walk_len
heads = args.heads
hidden_channels = args.hidden_channels
lr = args.lr
weight_decay = args.weight_decay

with torch.cuda.device('cuda:' + args.gpu_num):
    torch.cuda.empty_cache()

num_features = x.size(1)

torch.cuda.empty_cache()
print("Dimention of features after concatenation:", num_features)
set_random_seed(args.seed)

test_pos, test_neg, val_pos, val_neg = val_and_test

if args.model == 'gae':
    model = GAE(GCNEncoder(num_features, hidden_channels)).to(device)
elif args.model == 'vgae':
    model = VGAE(VariationalGCNEncoder(num_features, hidden_channels)).to(device)
else:
    out_channels=128
    encoder = Encoder(num_features, hidden_channels=64, out_channels=hidden_channels)
    discriminator = Discriminator(in_channels=hidden_channels, hidden_channels1=16,hidden_channels2=64, out_channels=out_channels)
    model = ARGVA(encoder, discriminator).to(device)

num_nodes,_=x.shape
n_train = int(edge_index.size(1) / 2)
edge_index = edge_index.to(device)

neg_edge_index = neg_edge_index.to(device)
x = x.to(device)

adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float).to(device)
neg_w_max = torch.zeros(num_nodes, num_nodes, dtype=torch.float).to(device)

if args.model != 'argva':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)#GAE,VGAE
else:
    #ARGVA
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    if args.data_name == 'pubmed':
        args.epoch_num=3000
    else:
        args.epoch_num=2000

deg_wei = torch.zeros(num_nodes, dtype=torch.float).to(device)

ave_deg = train_pos.size(1) * 2 / num_nodes
for i in range(num_nodes):
    deg_wei[i] = 2 / (train_pos.size(1)*2/adj2[i, i] + num_nodes) * num_nodes#harmonic average

best_val_auc = 0
best_tes_auc = 0
best_tes_ap = 0

for epoch in range(1, args.epoch_num + 1):
    model.train()
    if args.model != 'argva':
        optimizer.zero_grad()#GAE、VGAE
    else:
        encoder_optimizer.zero_grad()#ARGVA

    if args.model == 'gae':
        z, z_self = model.encode(x, edge_index)#GAE
    else:
        z = model.encode(x, edge_index)

    perm = torch.randperm(tneg_row.size(0))[:n_train]
    neg_edge_index = torch.stack([tneg_row[perm], tneg_col[perm]], dim=0).to(device)

    if args.model == 'argva':
        # ARGVA
        for i in range(5):
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

    if args.haw:
        loss, neg_edge_index = recon_mul_loss(z, train_pos, deg_wei, device, neg_edge_index=neg_edge_index)
    else:
        loss = model.recon_loss(z, train_pos, neg_edge_index=neg_edge_index)

    if args.model == 'vgae':
        loss = loss + (1 / num_nodes) * model.kl_loss()  # VGAE
    elif args.model == 'argva':
        # ARGVA
        loss = loss + model.reg_loss(z)
        loss = loss + (1 / num_nodes) * model.kl_loss()
    loss.backward()

    if args.model != 'argva':
        optimizer.step()
    else:
        encoder_optimizer.step()#ARGVA

    if epoch % 1 == 0:
        model.eval()
        if args.model == 'gae':
            z, _ = model.encode(x, edge_index)#GAE
        else:
            z = model.encode(x, edge_index)
        z = z.cpu().clone().detach()
        auc, _ = compute_scores(z, val_pos, val_neg)
        auctes, aptes = compute_scores(z, test_pos, test_neg)
        auctr, _ = compute_scores(z, edge_index, neg_edge_index)
        if auc > best_val_auc:
            best_val_auc = auc
            best_tes_auc = auctes
            best_tes_ap = aptes
            record_z = z.clone().detach()
        print(f'Setp: {epoch:03d} /{args.epoch_num:03d}, Loss : {loss.item():.4f}, Train_auc:{auctr:.4f}, Val_auc:{best_val_auc:.4f}, Val_aucnow:{auc:.4f}, Test_auc:{best_tes_auc:.4f}, Test_ap:{best_tes_ap:.4f}')

auc, ap = compute_scores(record_z, test_pos, test_neg)
print(f'{args.model} prediction accuracy, AUC: {auc:.4f}, AP: {ap:.4f}')
