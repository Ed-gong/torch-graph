#include <iostream>
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <iostream>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
//#include <torch/custom_class.h>
#include <string>
#include <vector>
//#include <cstdint>
using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;
#include "GCN.h"
#include "ManagerWrap.h"


//Gcn layer
GraphConv::GraphConv(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    //b = register_parameter("b", torch::randn(M));
}

//torch::Tensor forward(torch::Tensor input, ManagerWrap manager) 
torch::Tensor GraphConv::forward(torch::Tensor input, snap_t<dst_id_t>* snaph) 
{    
    torch::Tensor dst_data;

    if (N > M) {
    //mult W first to reduce the feature size for aggregation

        input = torch::matmul(input, W);
        dst_data = scatter_gather1(snaph, input, "sum");
    }
    else {
    //aggregate first then mult W
        dst_data = scatter_gather1(snaph, input, "sum");
        dst_data = torch::matmul(dst_data, W);

    }

    return dst_data;
}

//Gcn
GCN::GCN(int64_t in_features, int64_t hidden_size, int64_t num_class) 
        : conv1(in_features, hidden_size), conv2(hidden_size, num_class) 
{
    std::cout<< "initilize the GCN" << std::endl;
}

torch::Tensor GCN::forward(torch::Tensor input, snap_t<dst_id_t>* snaph) 
{
    torch:: Tensor h = conv1.forward(input, snaph);
    h = torch::relu(h);
    h = conv2.forward(h, snaph);
    return h;
}


vector<torch::Tensor> GCN::parameters(){
    std::vector<torch::Tensor> result;
    torch::Tensor para1 = conv1.W;
    torch::Tensor para2 = conv2.W;
    result.push_back(para1);
    result.push_back (para2);
    return result;

}
