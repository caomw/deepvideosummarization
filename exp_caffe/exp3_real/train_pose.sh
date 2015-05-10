#!/usr/bin/env sh

/usr0/home/shihenw/caffe_701/caffe/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 --weights=bvlc_reference_caffenet.caffemodel
