// An example of using the PyTorch C++ API to implement a custom forward and backward function

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>

#include "type.h"
#include "graph_view.h"


using torch::Tensor;
using at::Scalar;

using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;

using torch::autograd::variable_list;
using torch::autograd::tensor_list;

//using torch::autograd::as_variable;
//using torch::autograd::as_variable_ref;

using torch::autograd::compute_requires_grad;
using torch::autograd::collect_next_edges;
using torch::autograd::flatten_tensor_args;


struct MyPowBackward : public Node {
    // Public members that we use to store the forward pass, such that we can use it in gradient calculation
    SavedVariable self_;
    Scalar exponent_;

    // The following function is called during the backward pass
    variable_list apply(variable_list&& grads) override {
        std::cout << "-> Computing MyPow Backward!" << std::endl;
        
        // Our function had one output, so we only expect 1 gradient
        auto& grad = grads[0];
        // Grab the data out of the saved variable
        auto self = self_.unpack();
        double exponent = exponent_.toDouble();

        // Variable list to hold the gradients at the function's input variables
        variable_list grad_inputs(1); 

        // Do gradient computation for each of the inputs
        if (should_compute_output(0)) {
            auto grad_result = exponent != 0.0 ? grad * exponent * self.pow(exponent - 1) 
                    : torch::zeros_like(self);
            grad_inputs[0] = grad_result;
        }

        return grad_inputs;
    }

    // Apparently we need to manually handle destruction of SavedVaribles
    void release_variables() override {
        self_.reset_data();
        self_.reset_grad_function();
    }
};

torch::Tensor scatter_gather(torch::Tensor &input_feature, gview_t<dst_id_t>* snaph, string gather_operator){

//this function only perform the messge passing and gather function, we assume the input_future
//is the message each node need to pass to other nodes.

    //the input_message is the scatter messge

    //build the mailbox
    std::map<int, torch::Tensor> mailbox;

    vid_t v_count = snaph->get_vcount();
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;

    for (vid_t v = 0; v < v_count; v++) {
        if ( 0 == (nebr_count = snaph->get_nebrs_out(v, header))) continue;

         for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //If here is the first message for the node, build the dictionary
            if (mailbox.count(sid) == 0){
                mailbox[v] = input_feature[sid]; 
            }
            // If the node has already received some messages, we concatenate the received tensor
            torch::Tensor temp = torch::cat({mailbox[sid], input_feature[sid]}, 0);
            mailbox[sid] = temp;

            }

        }
    //gather procedure
    torch::Tensor temp; 
    for (std::map<int,torch::Tensor>::iterator it = mailbox.begin(); it!= mailbox.end(); ++it){
        
        if (gather_operator == "max"){
            temp = torch::max(it -> second);
        }
        if (gather_operator == "min"){
            temp = torch::min(it -> second);
        }
        if (gather_operator == "sum"){
            temp = torch::sum(it -> second);
         }
             it->second = temp;
        }

    // return a tensor with the input value after gather
    torch::Tensor result = torch::zeros({100000});
    for (vid_t v = 0; v < v_count; v++) {
        //loop the mailbox by key:node_id, value:tensor,
        //if the node did not reveive any message, that node will be 0
         map<int,torch::Tensor>::iterator it = mailbox.find(v);
         if (it == mailbox.end()) continue;
         result[v] = mailbox[v];
    }

    return result;
    

}


Tensor MyPowForward(torch::Tensor &input_feature, gview_t<dst_id_t>* snaph, string gather_operator)
{
    std::cout << "-> Computing MyPow Forward!" << std::endl;
    // we use the "max" as the gather operation here
    torch::Tensor tmp = scatter_gather(input_feature, snaph, "max"); //
    

    /*
Tensor MyPowForward(const Tensor & self, Scalar exponent) {
    std::cout << "-> Computing MyPow Forward!" << std::endl;
    // Compute the function's output
    //auto& self_ = as_variable_ref(self);
    auto tmp = self.data().pow(exponent); // compute the output based on the tensor's data
    

    auto result = tmp;

    // Prepare the infrastructure for computing the function's gradient
    if (compute_requires_grad( self )) {
        // Initialize the gradient function
        auto grad_fn = std::shared_ptr<MyPowBackward>(new MyPowBackward(), deleteNode);

        // Connect into the autograd graph
        grad_fn->set_next_edges(collect_next_edges( self ));

        // Save the function arguments for use in the backwards pass
        grad_fn->self_ = SavedVariable(self, false);
        grad_fn->exponent_ = exponent;

        // Attach the gradient function to the result
        set_history(flatten_tensor_args( result ), grad_fn);
    }

    return result;

    */
    return tmp;
}

/*
int main() {
    auto a = 3*torch::ones({3,3});
    a.set_requires_grad(true);

    std::cout << "Begin Forward Pass" << std::endl;
    / *
    auto b = MyPowForward(a, 2).sum();

    std::cout << "Begin Backward Pass" << std::endl;
    b.backward();

    std::cout << a.grad() << std::endl;

    * /
}*/



//PYBIND11_MODULE(dcgan, m) {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &MyPowForward, "LLTM forward");
}

