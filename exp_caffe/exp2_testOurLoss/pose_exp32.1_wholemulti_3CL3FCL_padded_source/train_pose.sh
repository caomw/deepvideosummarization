#!/usr/bin/env sh

/usr0/home/shihenw/caffe/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1
