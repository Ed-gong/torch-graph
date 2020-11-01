#include "SnapWrap.h"
#include "ManagerWrap.h"
#include "GCN.h"
#include "GAT.h"


short CorePin(int id)
{
    return 0;
}

graph* g = new graph();

struct GCNWrap : torch::CustomClassHolder {
  GCNWrap(int64_t in_features_dim, int64_t hidden_size, int64_t num_class) 
    :net(in_features_dim, hidden_size, num_class)
  {
  }

  torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph)
  {   
      //std::cout<<"wu"<<std::endl;
      torch::Tensor result = net.forward(input, snaph);
      return result;
  }

  vector<torch::Tensor> parameters(){
        vector<torch::Tensor> result = net.parameters();
        return result;
  }
  GCN net; 
};


struct GATWrap : torch::CustomClassHolder {
  GATWrap(int64_t in_dim, int64_t hidden_dim, int64_t out_dim) 
    :gat(in_dim, hidden_dim, out_dim){
  }

  torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrapW> snaph)
  {   
      torch::Tensor result = gat.forward(input, snaph);
      return result;
  }

  vector<torch::Tensor> parameters(){
      vector<torch::Tensor> result = gat.parameters();
      return result;
  }

  GAT gat;
};

template <class T>
void export_manager(torch::Library &m, std::string typestr) {
  std::string name = std::string("ManagerWrap") + typestr;
  m.class_<ManagerWrap_t<T>>(name)
    .def(torch::init<int64_t, int64_t, string>())
    .def("create_static_view",&ManagerWrap_t<T>::reg_static_view)
    .def("scatter_gather", &ManagerWrap_t<T>::scatter_gather)
    .def("adj_matrix", &ManagerWrap_t<T>::check_adjacency_matrix)
    ;
}

template <class T>
void export_static_view(torch::Library &m, std::string typestr) {
  std::string name = std::string("SnapWrap") + typestr;
  m.class_<SnapWrap_t<T>>(name)
    .def(torch::init<>())
  ;
}

TORCH_LIBRARY(my_classes, m) {
  export_static_view<dst_id_t>(m, "");
  export_static_view<weight_sid_t>(m, "W");

  export_manager<dst_id_t>(m, "");
  export_manager<weight_sid_t>(m, "W");

  m.class_<GCNWrap>("GCNWrap")
    .def(torch::init<int64_t, int64_t, int64_t>())
    .def("forward", &GCNWrap::forward)
    .def("parameters",&GCNWrap::parameters)
    //.def("get_current_graph",&GCNWrap::get_current_graph)
  ;
  m.class_<GATWrap>("GATWrap")
    .def(torch::init<int64_t, int64_t, int64_t>())
    .def("forward", &GATWrap::forward)
    .def("parameters",&GATWrap::parameters)
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

