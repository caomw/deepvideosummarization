#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  Forward_cpu(bottom, top);
  return;

  printf("Forward_gpu 0\n");

  int N=bottom[0]->num(),
      C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();
  int gtN=(int)bottom[1]->cpu_data()[0];
  
  printf("Forward_gpu 1\n");

  int inN=N-gtN;
  int CxHxW=C*H*W;
  caffe_gpu_set(inN,Dtype(0),sumexp1overL2dist_.mutable_gpu_data());

  printf("Forward_gpu 2\n");

  caffe_gpu_set(gtN*inN, Dtype(1), all1s_.mutable_gpu_data());

  printf("Forward_gpu 3\n");

  // compute diff and L2dist
  for(int i=0;i<inN;i++){
    for(int j=0;j<gtN;j++){
      caffe_gpu_sub(CxHxW,
        bottom[0]->gpu_data()+(gtN+i)*CxHxW, // f(i)
        bottom[0]->gpu_data()+j*CxHxW, // f(j)
        diff_.mutable_gpu_data()+(i*gtN+j)*CxHxW); 
      caffe_gpu_dot(CxHxW,
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        L2dist_.mutable_gpu_data()+i*gtN+j);
    }
  }

  printf("Forward_gpu 4\n");

  // compute exp(1/L2dist)
  caffe_gpu_div(gtN*inN,all1s_.gpu_data(),L2dist_.gpu_data(),exp1overL2dist_.mutable_gpu_data());

  printf("Forward_gpu 5\n");


  caffe_gpu_exp(gtN*inN,exp1overL2dist_.gpu_data(),exp1overL2dist_.mutable_gpu_data());

  printf("Forward_gpu 6\n");

  // compute sum(exp(1/L2dist))
  for(int i=0;i<inN;i++)
    for(int j=0;j<gtN;j++)
      sumexp1overL2dist_.mutable_gpu_data()[i]+=exp1overL2dist_.gpu_data()[i*gtN+j];
  Dtype LOSS=Dtype(0);

  printf("Forward_gpu 7\n");

  // for each input frame f(i)
  for(int i=0;i<inN;i++){
    // compute LOSSi
    Dtype LOSSi=Dtype(0);
    for(int j=0;j<gtN;j++)
      LOSSi+=L2dist_.gpu_data()[i*gtN+j]*exp1overL2dist_.gpu_data()[i*gtN+j]/sumexp1overL2dist_.gpu_data()[i];
    // add to total LOSS
    LOSS+=LOSSi;
  }
  top[0]->mutable_cpu_data()[0]=LOSS;

  printf("Forward_gpu 8\n");
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Backward_cpu(top, propagate_down, bottom);
  return;


  const Dtype loss_weight=top[0]->cpu_diff()[0];
  int N=bottom[0]->num(),
      C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();
  int gtN=(int)bottom[1]->cpu_data()[0];
  int inN=N-gtN;
  int CxHxW=C*H*W;
  caffe_gpu_set(N*CxHxW,Dtype(0),bottom[0]->mutable_gpu_diff());
  // ===== for each GT frame f(j) ===== 
  for(int j=0;j<gtN;j++){
    Dtype* bottom_diff_j=bottom[0]->mutable_gpu_diff()+j*CxHxW;
    for(int i=0;i<inN;i++){
      // compute 1st term
      caffe_gpu_axpby(CxHxW,
        Dtype(-2)*exp1overL2dist_.gpu_data()[i*gtN+j]/sumexp1overL2dist_.gpu_data()[i],
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_j);
      // compute 2nd term
      caffe_gpu_axpby(CxHxW,
        exp1overL2dist_.gpu_data()[i*gtN+j]*Dtype(2)/L2dist_.gpu_data()[i*gtN+j]/sumexp1overL2dist_.gpu_data()[i],
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_j);
      // compute 3rd term
      caffe_gpu_axpby(CxHxW,
        exp1overL2dist_.gpu_data()[i*gtN+j]*exp1overL2dist_.gpu_data()[i*gtN+j]*Dtype(-2)
          /L2dist_.gpu_data()[i*gtN+j]/sumexp1overL2dist_.gpu_data()[i]/sumexp1overL2dist_.gpu_data()[i],
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_j);
    }
  }
  // ===== for each input frame f(i) ===== 
  for(int i=0;i<inN;i++){
    Dtype* bottom_diff_i=bottom[0]->mutable_gpu_diff()+(gtN+i)*CxHxW;
    // compute 1st term
    for(int j=0;j<gtN;j++){
      caffe_gpu_axpby(CxHxW,
        Dtype(2)*exp1overL2dist_.gpu_data()[i*gtN+j]/sumexp1overL2dist_.gpu_data()[i],
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_i);
    }
    // compute 2nd term
    for(int j=0;j<gtN;j++){
      caffe_gpu_axpby(CxHxW,
        exp1overL2dist_.gpu_data()[i*gtN+j]*Dtype(-2)/L2dist_.gpu_data()[i*gtN+j]/sumexp1overL2dist_.gpu_data()[i],
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_i);
    }
    // compute 3rd term
    Dtype scale3rdterm=Dtype(0);
    for(int j=0;j<gtN;j++)
      scale3rdterm+=L2dist_.gpu_data()[i*gtN+j]*exp1overL2dist_.gpu_data()[i*gtN+j];
    scale3rdterm*=Dtype(-1)/sumexp1overL2dist_.gpu_data()[i]/sumexp1overL2dist_.gpu_data()[i];
    for(int j=0;j<gtN;j++){
      caffe_gpu_axpby(CxHxW,
        Dtype(-2)*exp1overL2dist_.gpu_data()[i*gtN+j]/L2dist_.gpu_data()[i*gtN+j]/L2dist_.gpu_data()[i*gtN+j]*scale3rdterm,
        diff_.gpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_i);
    }
  }
  caffe_gpu_scal(N*CxHxW,loss_weight,bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe