import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from graphviz import Digraph
from torchviz import make_dot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn import metrics


def accuracy(prediction, label):
    correct = 0 
    result = []
    for i in range(len(label)):
        #print("nani?")
        result.append(prediction[i])
        if (prediction[i] == label[i]):
            correct = correct + 1
    return correct / float(len(label)), result

torch.classes.load_library("build/libdcgan.so")
print(torch.classes.loaded_libraries)

graph_data_path = "/home/datalab/data/test2"
# the number of node in the graph
num_node = 34
input_feature_dim = 5
# the features have 5 dimensions
embed = nn.Embedding(num_node, input_feature_dim)

weight1 = torch.tensor([[ 1.0697, -0.1419,  0.0782, -1.0929,  1.1936],
        [-0.3677,  0.9840,  0.5001,  0.5691,  2.8748],
        [ 1.4907,  1.0489,  0.8279,  0.3542,  0.0923],
        [ 0.2266, -0.9223,  0.1327,  0.0550,  0.4363]
        [-1.9041, -0.9566,  0.1589, -0.4105, -0.7027],
        [-0.5105,  1.0170, -1.9042, -0.8973,  0.0485],
        [-1.3870, -2.6555,  0.6737,  0.5907, -1.4471],
        [-0.3273, -2.5961,  1.4094,  0.2120, -0.3798],
        [ 1.1695,  0.2795,  0.1744,  1.7339,  0.1272],
        [ 0.1529,  0.6807,  1.3838,  0.8028,  0.8778],
        [ 0.0781, -0.8352, -0.1175, -0.7997,  0.3524],
        [ 0.4686,  0.1325,  0.7024, -0.0676,  0.0301],
        [-0.8012,  1.6288, -1.3279,  0.5307, -0.5970],
        [ 0.0422, -0.8449,  0.8808,  0.6338, -0.4419],
        [-0.7238,  0.1553, -0.2273, -0.3242,  0.1480],
        [ 1.3768,  0.0718,  0.3425,  1.1562, -1.7056],
        [-0.0632,  0.1719, -0.6857,  0.2136,  0.2518],
        [-0.7727,  0.3532, -0.0727, -0.0459, -0.1422],
        [-0.1596, -1.4240, -0.5345,  1.0060, -0.1158],
        [ 0.4385, -0.0561,  0.8155, -0.3299, -0.6185],
        [ 0.9935, -1.7025,  1.9390, -0.3929,  0.2246],
        [ 0.6437, -0.4821, -0.6707,  1.4761,  0.9519],
        [ 0.8429,  0.4821,  1.2774, -0.4937, -0.3242],
        [ 0.1918, -0.3590, -1.5829, -0.7261, -0.2126],
        [-0.7885, -0.0722,  0.3902,  2.0317,  0.0518],
        [ 0.4079,  0.4579, -0.6410,  0.0372, -1.1750],
        [ 1.1448, -0.3493, -1.0203,  0.6427, -0.9110],
        [-0.2889,  2.0563,  0.9192,  0.0903,  0.5734],
        [ 0.6687, -0.6856,  1.3001,  0.4206,  0.4641],
        [ 0.7421,  1.2220,  0.7890,  1.0467, -0.3200],
        [-0.4786,  1.3458, -0.3373,  0.4424,  0.8443],
        [ 0.5896, -1.6171, -0.2082, -0.6664, -0.5781],
        [ 1.2394,  0.2473, -1.9055, -1.3474, -1.4021],
        [ 1.8865, -0.3820, -0.4517,  1.0323,  0.4815]], requires_grad=True)
    

embed = nn.Embedding.from_pretrained(weight1)
print ("input_feature")
print(embed.weight)
inputs = embed.weight
print(weight1.size())
print("???")
net = torch.classes.my_classes.GATWrap(input_feature_dim, 5, 2)
manager1 = torch.classes.my_classes.ManagerWrap(1, num_node, graph_data_path)
manager = torch.classes.my_classes.SnapWrap()
manager1.create_static_view(manager)


weight2 = torch.tensor([[-0.0202, -0.1091,  0.0603,  0.2008, -0.6365],
        [-0.1158, -0.4139, -0.4007, -0.3206, -0.3624],
        [-0.6508,  0.2236, -0.1202, -0.5056,  0.2951],
        [ 0.1367,  0.4756, -0.1852, -0.2253,  0.2737],
        [-0.3203,  0.4027,  0.6177, -0.4073,  0.0845]], requires_grad=True)
weight3 = torch.tensor([[-0.4980, -0.6419],
        [-0.0758, -0.3551],
        [-0.9030,  0.2837],
        [-0.4302, -0.8633],
        [-0.6338,  0.7542]], requires_grad=True)

#print ("1111")
#print (net.parameters())

#assign two labels to only two nodes
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different
# define the test for the remaining node
temp = []
for i in range(33):
    temp.append(i)
temp.remove(0)
labeled_nodes_test = torch.tensor(temp)
labels_test = torch.tensor([0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1])



# train the network
loss_record = []
accu_record = []

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
#optimizer = torch.optim.Adam([net.parameters(), itertools.chain(embed.parameters())], lr=0.01)
all_logits = []
for epoch in range(200):
    logits = net.forward(inputs, manager)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    
    loss = F.nll_loss(logp[labeled_nodes], labels)
    #make_dot(loss).render("torch_graph", format="png")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    loss_record.append(loss.item())
    #print ("the parameter after one update")
    logp_acc = torch.max(logp, 1).indices
    accu, labels_test_temp = accuracy(logp_acc[labeled_nodes_test],labels_test)
    accu_record.append(accu)
    print('Epoch %d | accuracy: %.4f' % (epoch, accu))
    print("evaluation report:")
    print(metrics.classification_report(labels_test, labels_test_temp, digits=3))

    
    #print (weight2)
    #print(weight3)
"""
# check the node predicton class
print ("node prediction class")
for v in range(34):
    temp = all_logits[199][v].numpy()
    cls = temp.argmax()
    print ("node" + str(v) + ":" + str(cls) + "\n")

print ("loss_record")
print(loss_record)
print("accu")
print(accu_record)
"""
