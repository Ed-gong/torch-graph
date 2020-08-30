import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.classes.load_library("build/libdcgan.so")
print(torch.classes.loaded_libraries)

graph_data_path = "/home/datalab/data/test1"
# the number of node in the graph
num_node = 100
input_feature_dim = 8
# the features have 5 dimensions
embed = nn.Embedding(num_node, input_feature_dim)
inputs = embed.weight
net = torch.classes.my_classes.GCNWrap(input_feature_dim, 5, 2)
manager = torch.classes.my_classes.ManagerWrap(1, num_node, graph_data_path)
manager.create_static_view(manager)

#assign two labels to only two nodes
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

# train the network
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
for epoch in range(50):
    logits = net.forward(inputs, manager)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    #optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
