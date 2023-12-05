import numpy as np
import torch
import dgl
import dgl.nn.pytorch as dglnn
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F



def create_hg(A, R, B, EOT, device):
    hetero_graph = dgl.heterograph({
    # connect A -> R
    ('object', 'link_or', 'relation'): ([0], [0]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> EOT, B -> EOT
    ('object', 'link_oe', 'eot'): ([0 ,1], [0, 0]),
    # R -> EOT
    ('relation', 'link_re', 'eot'): ([0], [0])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = EOT
    
    obj_features = torch.concat([A, B])
    R_features = R
    EOT_features = EOT
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}


def create_hg_self_loop(A, R, B, EOT, device):
    hetero_graph = dgl.heterograph({
    # connect A -> R
    ('object', 'link_or', 'relation'): ([0], [0]),
    # add self loop
    ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> EOT, B -> EOT
    ('object', 'link_oe', 'eot'): ([0 ,1], [0, 0]),
    # R -> EOT
    ('relation', 'link_re', 'eot'): ([0], [0])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = EOT
    
    obj_features = torch.concat([A, B])
    R_features = R
    EOT_features = EOT
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}

def create_hg_A_ARB_self_loop(A, R, B, A_EOT, ARB_EOT, device):
    # if add_noise_on_A:
    #     noise = torch.randn_like(A_EOT)
    #     A_EOT = A_EOT + 0.01*noise
    # return A and ARB 
    # EOT: 2, 1024, first one is A's second one is ARB's
    # connect A -> R
    hetero_graph = dgl.heterograph({
    # add self loop
    # ('eot', 'link_ee', 'eot'): ([1], [1]),
    ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
    ('object', 'link_or', 'relation'): ([0], [0]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> ARB EOT, B -> ARB EOT, A-> A EOT
    ('object', 'link_oe', 'eot'): ([0 ,1, 0], [1, 1, 0]),
    # R -> ARB EOT
    ('relation', 'link_re', 'eot'): ([0], [1])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = torch.concat([A_EOT, ARB_EOT])
    
    obj_features = torch.concat([A, B])
    R_features = R
    EOT_features = torch.concat([A_EOT, ARB_EOT])
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()} 
        h = self.conv2(graph, h)  
        return h
    

class RGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, heads=[2,2]):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, hid_feats, heads[0])
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hid_feats* heads[0], out_feats, heads[1])
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v).flatten(1) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: v.mean(1) for k, v in h.items()}
        
        return h