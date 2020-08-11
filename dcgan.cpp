// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>
#include <iostream>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <string>
#include <vector>
//#include <cstdint>

#include "type.h"
#include "graph.h"
#include "plain_to_edge.h"
#include "graph_view.h"


using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;

//using torch::autograd::as_variable;
//using torch::autograd::as_variable_ref;
//using at::Scalar;
//using torch::autograd::compute_requires_grad;
//using torch::autograd::collect_next_edges;
//using torch::autograd::flatten_tensor_args;

graph*g = new graph();
template <class T>
struct ManagerWrap : torch::CustomClassHolder {
  plaingraph_manager_t<T>* manager;

  ManagerWrap(int64_t flags) {
      manager = new plaingraph_manager_t<T>();
      manager -> schema(0);
      manager -> setup_graph(1000);
      manager -> prep_graph("/home/datalab/data/test1","");
      std:cout << "-> initilize the DAG!!" << std::endl;
  }
  
torch::Tensor scatter_gather(const torch::Tensor & input_feature, string gather_operator){
    snap_t<dst_id_t>* snaph = 0;
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
};


TORCH_LIBRARY(my_classes, m) {
  m.class_<ManagerWrap<dst_id_t>>("ManagerWrap")
  //m.class_<ManagerWrap<int64_t>>("ManagerWrap")
    // The following line registers the contructor of our MyStackClass
    // class that takes a single `std::vector<std::string>` argument,
    // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
    // Currently, we do not support registering overloaded
    // constructors, so for now you can only `def()` one instance of
    // `torch::init`.
    .def(torch::init<int64_t>())
    // The next line registers a stateless (i.e. no captures) C++ lambda
    // function as a method. Note that a lambda function must take a
    // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // as the first argument. Other arguments can be whatever you want.
    // The following four lines expose methods of the MyStackClass<std::string>
    // class as-is. `torch::class_` will automatically examine the
    // argument and return types of the passed-in method pointers and
    // expose these to Python and TorchScript accordingly. Finally, notice
    // that we must take the *address* of the fully-qualified method name,
    // i.e. use the unary `&` operator, due to C++ typing rules.
    .def("scatter_gather", &ManagerWrap<dst_id_t>::scatter_gather)
    //.def("scatter_gather", &ManagerWrap<int64_t>::scatter_gather)
    //.def("pop", &MyStackClass<std::string>::pop)
    //.def("clone", &MyStackClass<std::string>::clone)
    //.def("merge", &MyStackClass<std::string>::merge)
  ;
}









