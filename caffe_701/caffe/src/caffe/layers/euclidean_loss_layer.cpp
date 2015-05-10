#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


#define PENALTY_CONST 1

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int N=bottom[0]->num(),
      C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();
  int gtN=(int)bottom[1]->cpu_data()[0];
  int inN=N-gtN;
  
  //printf("reshape 0\n");
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  //printf("reshape 1\n");
  diff_.Reshape(gtN*inN,C,H,W);
  all1s_.Reshape(gtN*inN,1,1,1);
  L2dist_.Reshape(gtN*inN,1,1,1);
  exp1overL2dist_.Reshape(gtN*inN,1,1,1);
  sumexp1overL2dist_.Reshape(inN,1,1,1);
  
  gtdiff_.Reshape(gtN*gtN,C,H,W);
  gtL2dist_.Reshape(gtN*gtN,1,1,1);
  // printf("reshape 2\n");

  // printf("reshape 3\n");
  // printf("diff's dim: %d %d %d %d\n", diff_.num(), diff_.channels(), diff_.height(), diff_.width());
  // printf("all1s's dim: %d %d %d %d\n", all1s_.num(), all1s_.channels(), all1s_.height(), all1s_.width());
  // printf("l2dist's dim: %d %d %d %d\n", L2dist_.num(), L2dist_.channels(), L2dist_.height(), L2dist_.width());
  // printf("exp1overl2dist's dim: %d %d %d %d\n", exp1overL2dist_.num(), exp1overL2dist_.channels(), exp1overL2dist_.height(), exp1overL2dist_.width());
  // printf("sumexp1overl2dist's dim: %d %d %d %d\n", sumexp1overL2dist_.num(), sumexp1overL2dist_.channels(), sumexp1overL2dist_.height(), sumexp1overL2dist_.width());
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  int N=bottom[0]->num(),
      C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();
  int gtN=(int)bottom[1]->cpu_data()[0];
  int inN=N-gtN;
  int CxHxW=C*H*W;
  caffe_set(inN,Dtype(0),sumexp1overL2dist_.mutable_cpu_data());
  caffe_set(gtN*inN,Dtype(1),all1s_.mutable_cpu_data());
  caffe_set(gtN*inN*CxHxW,Dtype(0),diff_.mutable_cpu_data());
  caffe_set(gtN*gtN*CxHxW,Dtype(0),gtdiff_.mutable_cpu_data());
  
  // bool flag = 0;
  // //printf("outputing gradients: ");
  // for(int i=0; i<N*CxHxW; i++)
  //   if(isnan(diff_.cpu_data()[i])) {
  //      flag = 1;
  //      break;
  //   }
  //   //printf("%f ", bottom[0]->cpu_diff()[i]);
  // if(flag) printf("Forward diff storaged has nan\n");

  // compute diff and L2dist
  for(int i=0;i<inN;i++){
    for(int j=0;j<gtN;j++){
      caffe_sub(CxHxW,
        bottom[0]->cpu_data()+(gtN+i)*CxHxW, // f(i)
        bottom[0]->cpu_data()+j*CxHxW, // f(j)
        diff_.mutable_cpu_data()+(i*gtN+j)*CxHxW);

      L2dist_.mutable_cpu_data()[i*gtN+j]=caffe_cpu_dot(CxHxW,
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        diff_.cpu_data()+(i*gtN+j)*CxHxW);
    }
  }

  Dtype thre = 10000000000;
  // compute exp(1/L2dist)
  caffe_div(gtN*inN,all1s_.cpu_data(),L2dist_.cpu_data(),exp1overL2dist_.mutable_cpu_data());
  caffe_exp(gtN*inN,exp1overL2dist_.cpu_data(),exp1overL2dist_.mutable_cpu_data());
  
  for(int i=0;i<inN;i++){
    bool flag = 0;
    int index;
    for(int j=0;j<gtN;j++){
      if(exp1overL2dist_.cpu_data()[i*gtN+j] > thre || isinf(exp1overL2dist_.cpu_data()[i*gtN+j])){
        flag = 1;
        index = j;
        break;
      }
    }

    if(flag){
      for(int j=0;j<gtN;j++){
        if(j!=index) exp1overL2dist_.mutable_cpu_data()[i*gtN+j] = 0;
        else exp1overL2dist_.mutable_cpu_data()[i*gtN+j] = 1;
      }
    }
  }

  // compute sum(exp(1/L2dist))
  for(int i=0;i<inN;i++)
    for(int j=0;j<gtN;j++)
      sumexp1overL2dist_.mutable_cpu_data()[i]+=exp1overL2dist_.cpu_data()[i*gtN+j];
  
  Dtype LOSS=Dtype(0);
  // for each input frame f(i)
  for(int i=0;i<inN;i++){
    // compute LOSSi
    Dtype LOSSi=Dtype(0);
    for(int j=0;j<gtN;j++)
      LOSSi+=L2dist_.cpu_data()[i*gtN+j]*exp1overL2dist_.cpu_data()[i*gtN+j]/sumexp1overL2dist_.cpu_data()[i];
    // add to total LOSS
    LOSS+=LOSSi;
  }
  top[0]->mutable_cpu_data()[0]=LOSS/Dtype(inN);

  if(gtN > 1){
    // compute gtdiff_ and gtL2dist and LOSS_REG
    Dtype LOSS_REG=Dtype(0);
    for(int j=0;j<gtN;j++){
      for(int k=0;k<gtN;k++){
        if(k==j) continue;
        caffe_sub(CxHxW,
          bottom[0]->cpu_data()+j*CxHxW, // f(i)
          bottom[0]->cpu_data()+k*CxHxW, // f(j)
          gtdiff_.mutable_cpu_data()+(j*gtN+k)*CxHxW);

        gtL2dist_.mutable_cpu_data()[j*gtN+k]=caffe_cpu_dot(CxHxW,
          gtdiff_.cpu_data()+(j*gtN+k)*CxHxW,
          gtdiff_.cpu_data()+(j*gtN+k)*CxHxW);

        Dtype oneovergtL2dist=std::min(Dtype(1)/gtL2dist_.cpu_data()[j*gtN+k],Dtype(10000000000000));
        LOSS_REG+=oneovergtL2dist;
      }
    }
    top[0]->mutable_cpu_data()[0]+=Dtype(PENALTY_CONST)*LOSS_REG/(Dtype(gtN)*Dtype(gtN-1));
  }
  



  // Dtype maxValue = 0;
  // for(int i=0; i<gtN*inN; i++){
  //   if(exp1overL2dist_.cpu_data()[i] > maxValue){
  //     maxValue = exp1overL2dist_.cpu_data()[i];
  //   }
  // }
  // printf("forward: max value is %f\n", maxValue);

  // Dtype minValue = 1e9;
  // for(int i=0; i<inN; i++){
  //   if(sumexp1overL2dist_.cpu_data()[i] < minValue){
  //     minValue = sumexp1overL2dist_.cpu_data()[i];
  //   }
  // }
  // printf("forward: min value sum is %f\n", minValue);

  // flag = 0;
  // //printf("outputing gradients: ");
  // for(int i=0; i<N*CxHxW; i++)
  //   if(isnan(diff_.cpu_data()[i])) {
  //      flag = 1;
  //      break;
  //   }
  //   //printf("%f ", bottom[0]->cpu_diff()[i]);
  // if(flag) printf("Forward diff (3) storaged has nan\n");


  // minValue = 1e9;
  // for(int i=0; i<inN; i++){
  //   if(L2dist_.cpu_data()[i] < minValue){
  //     minValue = L2dist_.cpu_data()[i];
  //   }
  // }
  // printf("forward: min value L2dist_ is %f\n", minValue);

}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype loss_weight=top[0]->cpu_diff()[0];
  int N=bottom[0]->num(),
      C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();
  int gtN=(int)bottom[1]->cpu_data()[0];
  int inN=N-gtN;
  int CxHxW=C*H*W;
  caffe_set(N*CxHxW,Dtype(0),bottom[0]->mutable_cpu_diff());

  // bool flag = 0;
  // //printf("outputing gradients: ");
  // for(int i=0; i<N*CxHxW; i++)
  //   if(isnan(diff_.cpu_data()[i])) {
  //      flag = 1;
  //      break;
  //   }
  //   //printf("%f ", bottom[0]->cpu_diff()[i]);
  // if(flag) printf("diff storaged has nan\n");
  // printf("diff storaged: ");
  // for(int i=0; i<10; i++)
  //  printf("%f ", diff_.cpu_data()[i]);
  // printf("\n");
  Dtype thre = 1e-8;

  // ===== for each GT frame f(j) ===== 
  for(int j=0;j<gtN;j++){
    Dtype* bottom_diff_j=bottom[0]->mutable_cpu_diff()+j*CxHxW;
    for(int i=0;i<inN;i++){
      // compute 1st term
      caffe_cpu_axpby(CxHxW,
        Dtype(-2)*exp1overL2dist_.cpu_data()[i*gtN+j]/sumexp1overL2dist_.cpu_data()[i],
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_j);
      
      if(L2dist_.cpu_data()[i*gtN+j] < thre) continue;

      // compute 2nd term
      caffe_cpu_axpby(CxHxW,
        exp1overL2dist_.cpu_data()[i*gtN+j]*Dtype(2)/L2dist_.cpu_data()[i*gtN+j]/sumexp1overL2dist_.cpu_data()[i],
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_j);
      // compute 3rd term
      caffe_cpu_axpby(CxHxW,
        exp1overL2dist_.cpu_data()[i*gtN+j]*exp1overL2dist_.cpu_data()[i*gtN+j]*Dtype(-2)
          /L2dist_.cpu_data()[i*gtN+j]/sumexp1overL2dist_.cpu_data()[i]/sumexp1overL2dist_.cpu_data()[i],
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_j);
    }

  }

  // flag = 0;
  // //printf("outputing gradients: ");
  // for(int i=0; i<N*CxHxW; i++)
  //   if(isnan(bottom[0]->cpu_diff()[i])){
  //      flag = 1;
  //      break;
  //   }
  //   //printf("%f ", bottom[0]->cpu_diff()[i]);
  // if(flag) printf("outputing gradients (1) has nan\n");

  // ===== for each input frame f(i) ===== 
  for(int i=0;i<inN;i++){
    Dtype* bottom_diff_i=bottom[0]->mutable_cpu_diff()+(gtN+i)*CxHxW;
    Dtype scale3rdterm=Dtype(0);
    for(int j=0;j<gtN;j++)
      scale3rdterm+=L2dist_.cpu_data()[i*gtN+j]*exp1overL2dist_.cpu_data()[i*gtN+j];
    scale3rdterm*=Dtype(-1)/sumexp1overL2dist_.cpu_data()[i]/sumexp1overL2dist_.cpu_data()[i];

    
    for(int j=0;j<gtN;j++){
      // compute 1st term
      caffe_cpu_axpby(CxHxW,
        Dtype(2)*exp1overL2dist_.cpu_data()[i*gtN+j]/sumexp1overL2dist_.cpu_data()[i],
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_i);
      
      if(L2dist_.cpu_data()[i*gtN+j] < thre) continue;

      // compute 2nd term
      caffe_cpu_axpby(CxHxW,
        exp1overL2dist_.cpu_data()[i*gtN+j]*Dtype(-2)/L2dist_.cpu_data()[i*gtN+j]/sumexp1overL2dist_.cpu_data()[i],
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_i);
    
      // compute 3rd term
      caffe_cpu_axpby(CxHxW,
        Dtype(-2)*exp1overL2dist_.cpu_data()[i*gtN+j]/L2dist_.cpu_data()[i*gtN+j]/L2dist_.cpu_data()[i*gtN+j]*scale3rdterm,
        diff_.cpu_data()+(i*gtN+j)*CxHxW,
        Dtype(1),bottom_diff_i);
    }
  }
  caffe_scal(N*CxHxW, loss_weight/Dtype(inN), bottom[0]->mutable_cpu_diff());


  if(gtN>1){
    // compute regularization term
    for(int j=0;j<gtN;j++){
      Dtype* bottom_diff_j=bottom[0]->mutable_cpu_diff()+j*CxHxW;
      for(int k=0;k<gtN;k++){
        // compute regularization term
        if(k==j) continue;
        Dtype oneovergtL2dist=std::min(Dtype(1)/gtL2dist_.cpu_data()[j*gtN+k],Dtype(10000000000000));
        caffe_cpu_axpby(CxHxW,
          Dtype(-2)*oneovergtL2dist*oneovergtL2dist*Dtype(PENALTY_CONST)/(Dtype(gtN)*Dtype(gtN-1))*loss_weight,
          gtdiff_.cpu_data()+(j*gtN+k)*CxHxW,
          Dtype(1),bottom_diff_j);
      }
    }
  }

  // flag = 0;
  // //printf("outputing gradients: ");
  // for(int i=0; i<N*CxHxW; i++)
  //   if(isnan(bottom[0]->cpu_diff()[i])){
  //      flag = 1;
  //      break;
  //   }
  //   //printf("%f ", bottom[0]->cpu_diff()[i]);
  // if(flag) printf("outputing gradients (2) has nan\n");
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe   