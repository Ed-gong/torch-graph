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

#include "SnapWrap.h"
#include "ManagerWrap.h"
#include "GCN.h"
 
/*
c10::intrusive_ptr<SnapWrap> get_current_graph(c10::intrusive_ptr<ManagerWrap> manager, 
                c10::intrusive_ptr<SnapWrap> snaph)
{
    snap_t<dst_id_t>* snaph1 = net.get_current_graph(manager->manager,snaph->snaph);
    return snaph1;
}*/

struct GCNWrap : torch::CustomClassHolder {
  //plaingraph_manager_t<T>* manager;

  GCNWrap(int64_t in_features_dim, int64_t hidden_size, int64_t num_class) 
  :net(in_features_dim, hidden_size, num_class){
  }

  torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph)
  {
      torch::Tensor result = net.forward(input, snaph);
      return result;
  }

  vector<torch::Tensor> parameters(){
    /*
        for (const auto& p : net.parameters()) {
            std::cout << p << std::endl;
        }
        return net.parameters();
        */
        vector<torch::Tensor> result = net.parameters();
        return result;
  }
  GCN net; 
};

TORCH_LIBRARY(my_classes, m) {
  m.class_<SnapWrap>("SnapWrap")
    .def(torch::init<>())
  ;

  m.class_<ManagerWrap>("ManagerWrap")
    .def(torch::init<int64_t, int64_t, string>())
    .def("create_static_view",&ManagerWrap::create_static_view)
    .def("scatter_gather", &ManagerWrap::scatter_gather)
  ;
  m.class_<GCNWrap>("GCNWrap")
    .def(torch::init<int64_t, int64_t, int64_t>())
    .def("forward", &GCNWrap::forward)
    .def("parameters",&GCNWrap::parameters)
    //.def("get_current_graph",&GCNWrap::get_current_graph)
  ;
  //m.class_<ManagerWrap<int64_t>>("ManagerWrap")
    // The following line registers the contructor of our MyStackClass
    // class that takes a single `std::vector<std::string>` argument,
    // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
    // Currently, we do not support registering overloaded
    // constructors, so for now you can only `def()` one instance of
    // `torch::init`.
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
    //.def("get_vcount", &ManagerWrap<dst_id_t>::get_node)
    //.def("scatter_gather", &ManagerWrap<int64_t>::scatter_gather)
    //.def("pop", &MyStackClass<std::string>::pop)
    //.def("clone", &MyStackClass<std::string>::clone)
    //.def("merge", &MyStackClass<std::string>::merge)
}

