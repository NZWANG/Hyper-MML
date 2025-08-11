import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch_scatter import scatter_add
from torch.nn import Parameter
from typing import Optional
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree, softmax


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(HyperGCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph

        #self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
        #                       dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
        #                       return_feature=return_feature, use_residue=use_residue)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        #nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), highConv(nhidden, nhidden))
        #self.conv = highConv(nhidden, nhidden)

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])


        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   
        #out1 = self.reverse_features(dia_len, out1)                                     

        #---------------------------------------
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        #---------------------------------------
        return out1
        


    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 =[]
        index2 =[]
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i+3):
                if _ < 3:
                    index2 = index2 + [edge_count]*i
                else:
                    index2 = index2 + [edge_count]*3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            
            Gnodes=[]
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_,2))
                tmp = tmp + perm
            batch = batch + [batch_count]*i*3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1]*i + [0]*3

            node_count = node_count + i*num_modality

        index1 = torch.LongTensor(index1).view(1,-1)
        index2 = torch.LongTensor(index2).view(1,-1)
        hyperedge_index = torch.cat([index1,index2],dim=0).cuda() 
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0,max_node+1,1).repeat_interleave(2).view(1,-1),
                            torch.arange(max_edge+1,max_edge+1+max_node+1,1).repeat_interleave(2).view(1,-1)],dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1,1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()

        return edge_index, features


class highConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(highConv, self).__init__(aggr='add')  # "Add" aggregation.
        #self.lin = torch.nn.Linear(in_channels, out_channels)
        self.gate = torch.nn.Linear(2*in_channels, 1)
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # x_j e.g.[135090, 512]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        #h2 = x_i - x_j
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))#e.g.[135090, 1]

        return norm.view(-1, 1) * (x_j) *alpha_g

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out


class HypergraphConv(MessagePassing):
    """Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0,  **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.edgefc = torch.nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)


    def forward(self, x, hyperedge_index,
                hyperedge_weight = None,
                hyperedge_attr = None, EW_weight = None, dia_len = None):
        """
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        #if hyperedge_attr is not None:
        #    hyperedge_attr = self.edgefc(hyperedge_attr)
        #x = torch.matmul(x, self.weight)
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)      #[num_nodes]
        D = 1.0 / D                                                         # all 0.5 if hyperedge_weight is None
        D[D == float("inf")] = 0
        if EW_weight is None:
            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        else:
            B = scatter_add(EW_weight[hyperedge_index[0]],
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        B = 1.0 / B
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,#hyperedge_attr[hyperedge_index[1]],  
                             size=(num_nodes, num_edges))                   #num_edges,1,100
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))                   #num_nodes,1,100
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        
        return torch.nn.LeakyReLU()(out)  #


    def message(self, x_j, norm_i, alpha):
        H, F = self.heads, self.out_channels

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j      
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    #def update(self, aggr_out):
    #    return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
