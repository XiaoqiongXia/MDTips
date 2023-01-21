# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:27:38 2022

@author: toxicant
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size) #每个词和位置先随机初始化
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size) # 同一个向量进行3次线性变换得到query，key和value
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask # 将attention score + masking

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states

class transformer(nn.Sequential):
    def __init__(self):
        super(transformer, self).__init__()
        self.emb = Embeddings(vocab_size=4114, hidden_size=64, max_position_size=545, dropout_rate=0.1)
        self.encoder = Encoder_MultipleLayers(n_layer=2, hidden_size=64,intermediate_size=256,num_attention_heads=4,
                                                attention_probs_dropout_prob=0.1,hidden_dropout_prob=0.1)

    ### parameter v (tuple of length 2) is from utils.drug2emb_encoder 
    def forward(self, v_p,mask):
        e = v_p
        e_mask = mask# (batch_size,dim)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2) # (batch_size,1,1,dim)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0 # 将padding部分变成一个很负的值

        emb = self.emb(e) 
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:,0]

class DGL_NeuralFP(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/gat_predictor.py
    def __init__(self, in_feats, hidden_feats=None, max_degree = None, activation=None, predictor_hidden_size = None, predictor_activation = None, predictor_dim=None):
        super(DGL_NeuralFP, self).__init__()
        from dgllife.model.gnn.nf import NFGNN
        from dgllife.model.readout.sum_and_max import SumAndMax
        self.gnn = NFGNN(in_feats=in_feats,
                        hidden_feats=hidden_feats,
                        max_degree=max_degree,
                        activation=activation
                        )
        gnn_out_feats = self.gnn.gnn_layers[-1].out_feats
        self.node_to_graph = nn.Linear(gnn_out_feats, predictor_hidden_size)
        self.predictor_activation = predictor_activation
        self.readout = SumAndMax()
        self.transform = nn.Linear(predictor_hidden_size * 2, predictor_dim)
    def forward(self, bg):
        bg = bg.to(device)
        feats = bg.ndata.pop('h') 
        node_feats = self.gnn(bg, feats)
        node_feats = self.node_to_graph(node_feats)
        graph_feats = self.readout(bg, node_feats)
        graph_feats = self.predictor_activation(graph_feats)
        return self.transform(graph_feats)

class DGL_AttentiveFP(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/attentivefp_predictor.py#L17
    def __init__(self, node_feat_size, edge_feat_size, num_layers = 2, num_timesteps = 2, graph_feat_size = 200, predictor_dim=None):
        super(DGL_AttentiveFP, self).__init__()
        from dgllife.model.gnn import AttentiveFPGNN
        from dgllife.model.readout import AttentiveFPReadout
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps)
        self.transform = nn.Linear(graph_feat_size, predictor_dim)
    def forward(self, bg):
        bg = bg.to(device)
        node_feats = bg.ndata.pop('h')
        edge_feats = bg.edata.pop('e')
        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = self.readout(bg, node_feats, False)
        return self.transform(graph_feats)

class Trans_Fusion(nn.Sequential):
    def __init__(self):
        super(Trans_Fusion, self).__init__()
        self.encoder = Encoder_MultipleLayers(n_layer=3, hidden_size=256,intermediate_size=128,num_attention_heads=4,
                                              attention_probs_dropout_prob=0.1,hidden_dropout_prob=0.1)
    def forward(self,v,mask):
        encoded_layers = self.encoder(v, mask)
        return encoded_layers[:,0]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, v):
        v = self.linear(v)
        v = self.dropout(v)
        v = self.act(v)
        return v


class DTI(nn.Sequential):
    def __init__(self, **config):
        super(DTI, self).__init__()
        self.config = config
        self.model_drug = DGL_AttentiveFP(node_feat_size = 39, 
                                            edge_feat_size = 11,  
                                            num_layers = 3, 
                                            num_timesteps = 2, 
                                            graph_feat_size = 64, 
                                            predictor_dim = 256)
        self.model_protein = transformer()
        self.S_pro = MLP(320,256)
        self.K_pro = MLP(400,256)
        self.E_pro = MLP(1956,256)
        self.LayerNorm = LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        self.fusion = Trans_Fusion()
        self.hidden_dims = [1024,1024,512]
        layer_size = len(self.hidden_dims) + 1
        if config['data_type'] == 'MDTips':
            dims = [768] + self.hidden_dims + [1]
        if (config['data_type'] == 'Structure_KG') | (config['data_type'] == 'Structure_Expr') | (config['data_type'] == 'KG_Expr'):
            dims = [512] + self.hidden_dims + [1]
        if (config['data_type'] == 'Structure') | (config['data_type'] == 'Expr') | (config['data_type'] == 'KG'):
            dims = [256] + self.hidden_dims + [1]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
    def forward(self, data):
        # each encoding
        if self.config['data_type'] == 'MDTips':
            d_s = self.model_drug(data[0])
            p_s = self.model_protein(data[1],data[2])
            v_s = torch.cat((d_s, p_s), 1)
            v_s = self.S_pro(v_s)
            v_s = self.LayerNorm(v_s)
            v_kg = torch.cat((data[3], data[4]), 1)
            v_kg = self.K_pro(v_kg)
            v_kg = self.LayerNorm(v_kg)
            v_expr = torch.cat((data[5], data[6]), 1)
            v_expr = self.E_pro(v_expr)
            v_expr = self.LayerNorm(v_expr)
            v = torch.cat((v_s,v_kg,v_expr),1)
            
        if self.config['data_type'] == 'Structure_KG':
            d_s = self.model_drug(data[0])
            p_s = self.model_protein(data[1],data[2])
            v_s = torch.cat((d_s, p_s), 1)
            v_s = self.S_pro(v_s)
            v_s = self.LayerNorm(v_s)
            v_kg = torch.cat((data[3], data[4]), 1)
            v_kg = self.K_pro(v_kg)
            v_kg = self.LayerNorm(v_kg)
            v = torch.cat((v_s,v_kg),1)
            
        if self.config['data_type'] == 'Structure_Expr':
            d_s = self.model_drug(data[0])
            p_s = self.model_protein(data[1],data[2])
            v_s = torch.cat((d_s, p_s), 1)
            v_s = self.S_pro(v_s)
            v_expr = torch.cat((data[3], data[4]), 1)
            v_expr = self.E_pro(v_expr)
            v = torch.cat((v_s,v_expr),1)
            
        if self.config['data_type'] == 'KG_Expr':
            v_kg = torch.cat((data[0], data[1]), 1)
            v_kg = self.K_pro(v_kg)
            v_expr = torch.cat((data[2], data[3]), 1)
            v_expr = self.E_pro(v_expr)
            v = torch.cat((v_kg,v_expr),1)
        
        if self.config['data_type'] == 'Structure':
            d_s = self.model_drug(data[0])
            p_s = self.model_protein(data[1],data[2])
            v_s = torch.cat((d_s, p_s), 1)
            v = self.S_pro(v_s)
        
        if self.config['data_type'] == 'KG':
            v_kg = torch.cat((data[0], data[1]), 1)
            v = self.K_pro(v_kg)
        
        if self.config['data_type'] == 'Expr':
            v_expr = torch.cat((data[0], data[1]), 1)
            v = self.E_pro(v_expr)
            
        for i, l in enumerate(self.predictor):
            if i==(len(self.predictor)-1):
                v = l(v)
            else:
                v = F.relu(self.dropout(l(v)))
        return v























