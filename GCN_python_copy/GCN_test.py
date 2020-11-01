import torch
import torch.nn as nn
import graphconv as graphc
import util
import itertools
import torch.nn.functional as F
import pygraph as gone

class GCN(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(graphc.GraphConv(in_feats = in_feats, out_feats = n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(graphc.GraphConv(in_feats = n_hidden, out_feats = n_hidden))
        # output layer
        self.layers.append(graphc.GraphConv(in_feats = n_hidden, out_feats = n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            #if i != 0:
                #h = self.dropout(h)
            h = layer(self.graph, h)
            #print (h)
        return h


if __name__ == "__main__":
    ingestion_flag = gone.enumGraph.eUdir
    ifile = "/home/datalab/data/test3"
    num_node = 2708
    manager = gone.pgraph_manager_t(ingestion_flag, num_node, ifile, "")
    manager.run_bfs(1);
    snap_t = manager.create_static_view(gone.enumView.eStale)
    input_feature_dim = 1433
    net = GCN(snap_t, input_feature_dim, 16, 7, 3, 1)

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
    print(input_X)
    print("-------------------")
    for epoch in range(100):
        logits = net(input_X)
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[labeled_nodes_train], labels[labeled_nodes_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))

        # check the accuracy for test data
        logits_test = net.forward(input_X)
        logp_test = F.log_softmax(logits_test, 1)

        acc_val = util.accuracy(logp_test[labeled_nodes_test], labels[labeled_nodes_test])
        print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))

#
