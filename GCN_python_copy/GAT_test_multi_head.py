import torch
import torch.nn as nn
import gat_multi_head as graphgat
import util
import itertools
import torch.nn.functional as F
import pygraph as gone
import numpy as np
import datetime

class GAT(nn.Module):
    def __init__(self,
                 graph,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation = None,
                 feat_drop = 0.,
                 attn_drop = 0.,
                 negative_slope =0.2,
                 residual = False):
        super(GAT, self).__init__()
        self.graph = graph
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(graphgat.GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(graphgat.GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(graphgat.GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.graph, h).flatten(1)
        #print('111')
        #print(h.size())
        # output projection
        logits = self.gat_layers[-1](self.graph, h).mean(1)
        #print('debug')
        #print(logits.size())
        return logits


if __name__ == "__main__":
    ingestion_flag = gone.enumGraph.eUnidir | gone.enumGraph.eDoubleEdge| gone.enumGraph.eCreateEID
    ifile = "/home/datalab/data/test3/cora"
    num_node = 2708
    num_sources = 1
    num_thread = 2
    # manager = gone.pgraph_manager_tW(ifile, "", ingestion_flag, num_node)
    # snap_t = manager.create_static_view(gone.enumView.eStale)

    edge_dt = np.dtype([('src', np.int32), ('dst', np.int32), ('edgeid', np.int32)])

    outdir = ""
    graph = gone.init(1, 1, outdir, num_sources, num_thread)  # Indicate one pgraph, and one vertex type
    tid0 = graph.init_vertex_type(2708, True, "gtype") # initiate the vertex type
    pgraph = graph.create_schema(ingestion_flag, tid0, "friend", edge_dt) # initiate the pgraph

    # creating graph directly from file requires some efforts. Hope to fix that later
    manager = graph.get_pgraph_managerW(0) # This assumes single weighted graph, edgeid is the weight
    manager.add_edges_from_dir(ifile, ingestion_flag)  # ifile has no weights, edgeid will be generated
    pgraph.wait()  # You can't call add_edges() after wait(). The need of it will be removed in future.
    manager.run_bfs(1)

    snap_t = gone.create_static_view(pgraph, gone.enumView.eStale)


    input_feature_dim = 1433
    num_heads = 3
    num_layers = 3
    heads_array = ([num_heads] * num_layers) + [1]
    net = GAT(snap_t, num_layers, input_feature_dim, 16, 7, heads_array, activation = None, feat_drop = 0., attn_drop = 0., negative_slope =0.2, residual = False)

    labels, node_id, input_X = util.read_data()
    train_idx, val_idx, test_idx = util.limit_data(labels)
    input_train, input_test, output_train, output_test = util.get_train_test_data(train_idx, test_idx, input_X, labels)

    label_set = set(labels)
    class_label_list = []
    for each in label_set:
        class_label_list.append(each)
    labeled_nodes_train = torch.tensor(train_idx)  # only the instructor and the president nodes are labeled
    # labels_train = torch.tensor(output_train_label_encoded )  # their labels are different
    labeled_nodes_test = torch.tensor(test_idx)  # only the instructor and the president nodes are labeled
    # labels_test = torch.tensor(output_test_label_encoded)  # their labels are different
    # train the network
    optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr=0.01, weight_decay=5e-4)
    all_logits = []
    start = datetime.datetime.now()
    for epoch in range(200):
        logits = net(input_X)
        #print('hhhhha')
        #print(logits.size())
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        #print('loss_size', logp[labeled_nodes_train].size(), labels[labeled_nodes_train].size())
        loss = F.nll_loss(logp[labeled_nodes_train], labels[labeled_nodes_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))

        # check the accuracy for test data
        logits_test = net.forward(input_X)
        logp_test = F.log_softmax(logits_test, 1)

        acc_val = util.accuracy(logp_test[labeled_nodes_test], labels[labeled_nodes_test])
        #print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))

    end = datetime.datetime.now()
    difference = end - start
    print("the time of graphpy is:", difference)
    print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))


#


