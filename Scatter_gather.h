//
// Created by yidong on 8/31/20.
//

#ifndef UNTITLED1_SCATTER_GATHER_H
#define UNTITLED1_SCATTER_GATHER_H

#include <torch/torch.h>
#include "SnapWrap.h"

#ifndef _OPENMP
#define _OPENMP
#endif
//#include <ATen/ParallelOpenMP.h>
#include <ATen/Parallel.h>

using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;

//2D tensor
template <class T>
struct array2d_t {
    T* data_ptr;
    int64_t row_count;
    int64_t col_count;
    T* operator[] (int64_t index) {//returns a row
        return data_ptr + col_count*index;
    }
    array2d_t(T* a_ptr, int64_t a_row_count, int64_t a_col_count) {
        data_ptr = a_ptr;
        row_count = a_row_count;
        col_count = a_col_count;
    }
    void row_copy(T* ptr, int64_t index) {
        T* row_ptr = data_ptr + col_count*index;
        memcpy(row_ptr, ptr, col_count*sizeof(T)); 
    }
    void row_copy_norm(T* ptr, int64_t index, int degree) {
        T* row_ptr = data_ptr + col_count*index;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] = ptr[i]/degree;
        }
    }
    void row_add(T* ptr, int64_t index) {
        T* row_ptr = data_ptr + col_count*index;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] += ptr[i];
        }
    }
    void row_normalize(int64_t index, T degree) {
        T* row_ptr = data_ptr + col_count*index;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] /= degree;
        }
    }
};

//1D tensor
template <class T>
struct array1d_t {
    T* data_ptr;
    int64_t col_count;
    T operator[] (int64_t index) {//returns the element 
        return data_ptr[index];
    }
    array1d_t(int64_t a_col_count) {
        data_ptr = (T*)calloc(sizeof(T), a_col_count);
        col_count = a_col_count;
    }
    ~array1d_t() {
        free(data_ptr);
    }
    void copy(T* ptr) {
        memcpy(data_ptr, ptr, col_count*sizeof(T)); 
    }
    void add(T* ptr) {
        T* row_ptr = data_ptr;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] += ptr[i];
        }
    }
};


template <class T>
void scatter_gather1(snap_t<T>* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     string gather_operator, int64_t reverse, bool norm = true)
{
    vid_t v_count = snaph->get_vcount();
    int output_dim = input.col_count;

    //If in backward, normalize it first
    if (reverse == 1 && norm == true) {
        at::parallel_for(0, v_count, 0, [&](int64_t start, int64_t end) {
        for (vid_t v = start; v < end; v++) {
            degree_t degree = snaph->get_degree_in(v);
            if (degree > 1) {
                input.row_normalize(v, degree); 
            }
        }
        });
    }
    
    
    //Start of parallelism
    at::parallel_for(0, v_count, 0, [&](int64_t start, int64_t end) {
    for (vid_t v = start; v < end; v++)
    //for (vid_t v = 0; v < v_count; v++) 
    { 
        degree_t nebr_count = 0;
        degree_t degree = 0;
        vid_t sid;
        nebr_reader_t<T> header;
        if (reverse == 0) {
            nebr_count = snaph->get_nebrs_in(v, header);
            degree = snaph->get_degree_out(v);
        } else {
            nebr_count = snaph->get_nebrs_out(v, header);
            degree = snaph->get_degree_in(v);
        }
        // if one node do not have any neighbor, we do not scatter it's message
        if (nebr_count == 0) {
            //result[v] = input_feature[v];
            continue; 
        }
        
        // the node j scatter it's message to all neighors
        array1d_t<float> message(output_dim);//zero initialized
        //edit here for self loop
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            message.add(input[sid]);
        }
        //output.row_copy(message.data_ptr, v);
        //If in forward, normalize it now.
        if (degree > 1 && reverse ==0 && norm == true) {
            output.row_copy_norm(message.data_ptr, v, degree);
        } else {
            output.row_copy(message.data_ptr, v);
        }
    
    }
    });//end of parallelism
}


#endif //UNTITLED1_SCATTER_GATHER_H

