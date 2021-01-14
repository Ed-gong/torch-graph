import torch as th
from torch import nn





import sparse

# pylint: enable=W0235
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._in_feats_size = in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.linear = nn.Linear(
            self._in_feats_size, out_feats, bias=False)
        self.fc = nn.Linear(
            self._in_feats_size, out_feats, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        # re-initilize the parameter for linear layer
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        # re-initilize the parameter for attention layer
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        # re-initilize the parameter for linear layer
        # if isinstance(self.res_fc, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        self.fc_src, self.fc_dst = self.fc, self.fc
        feat_src = feat_dst = self.fc(feat).view(-1, self._out_feats)
        #print(feat_src.size())
        #print(self.attn_l.size())
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        #print(el.size())
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)



        # map_input = self.linear(feat).view(-1, self._num_heads, self._out_feats)
        # print (map_input.size())
        # print(self.attn_l.size())
        # el = th.matmul(map_input, self.attn_l)
        # er = th.matmul(map_input,  self.attn_r)

        num_vcount = graph.get_vcount()
        # since the score is scalar
        dim = feat_src.size(1)
        # apply_edge add the attention_score by vertex
        edge_score = sparse.apply_edge(graph, el, er)
        efficient_score = self.leaky_relu(edge_score)
        edge_score_by_softmax = sparse.edge_softmax(graph, efficient_score, num_vcount, dim)
        #     // std::cout << "nani6" << std::endl;
        # torch::Tensor
        rst = sparse.run_gspmv_op(graph, feat_src, edge_score_by_softmax, num_vcount, dim)
        #print (rst.size())

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
