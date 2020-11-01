import torch as th
import torch.utils.dlpack
import numpy as np
import ctypes as C
from ctypes.util import find_library
import pygraph as gone

#
# __all__ = ['gspmm', 'gsddmm', 'edge_softmax']
#


class GSpmv(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, X, norm, num_vcount, dim):
        #feat = X.clone()
        #feat = X.detach().numpy()
        #feat = X.data.cpu().numpy().copy()
        #feat = np.ctypeslib.as_array(X,shape=(num_vcount,dim))
        #print(X)
        feat = th.utils.dlpack.to_dlpack(X)

        #declare the output tensor here
        res = th.zeros(num_vcount, dim)
        #result = gone.array2d_t(re.data_ptr, num_vcount, dim) 
        #result = res.numpy()
        result = th.utils.dlpack.to_dlpack(res)

        graph.gspmv(feat, result, 0, norm) # do not specify the reduce operation

        ctx.backward_cache = graph, norm, num_vcount, dim
        return res

    @staticmethod
    def backward(ctx, dZ):
        graph, norm, num_vcount, dim = ctx.backward_cache
        #X = dZ.detach().numpy()
        #X = dZ.data.cpu().numpy().copy()
        #X = np.ctypeslib.as_array(dZ, shape=(num_vcount, dim))
        #X1 = dZ.clone()
        #X1 = X1.detach().numpy()
        X = th.utils.dlpack.to_dlpack(dZ)

        # todo convert the dZ to 2d array
        #dt = np.dtype([('src', np.float32), ('dst', np.float32)])
        res =  th.zeros(num_vcount, dim)
        #result = res.numpy() #np.zeros((num_vcount, dim), dtype = np.float32)
        result = th.utils.dlpack.to_dlpack(res)
        
        graph.gspmv(X, result, 1, norm)
        #res = th.from_numpy(result)

        return None, res, None, None, None





def run_gspmm(graph, X, norm, num_vcount, dim):
    return GSpmv.apply(graph, X, norm, num_vcount, dim)




