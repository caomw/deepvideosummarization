function [scores, maxlabel] = matcaffe_demo(model_file)
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 


%init caffe network (spews logging info)
%if exist('use_gpu', 'var')
%  matcaffe_init(use_gpu);
%else
addpath('/usr0/home/shihenw/');
matcaffe_init(1, '../pose_deploy.prototxt', model_file);
%end


  % For demo purposes we will use the peppers image
try
    load('/usr0/home/shihenw/testingImg/te1.mat');
catch
    im = imread('/usr0/home/shihenw/capture/140724-pose-sabih-vga-img/00001831_01_07.jpg');
    %imshow(im);

    % try
    %     matlabpool close;
    % end
    % matlabpool open 10;
    patches = generatePatches(im, 64, 8);
end

sizeOfPatches = size(patches);
numPatches = numel(patches);
scores = [];

data = cell2mat(permute(reshape(patches, [numel(patches) 1]), [4 2 3 1])); %into a 4D array

batch = 512;
numImages = size(data, 4);
numBatches = ceil(numImages/batch);
numPadding = batch * numBatches - numImages;
zeroPad = zeros(size(data,1), size(data,2), size(data,3), numPadding);
data_pad = cat(4, data, zeroPad);

%C{i} = zeros(2,2);

%getScores_batch(128);
for j = 1:numBatches
        im_batch = data_pad(:,:,:,batch*(j-1)+1:batch*j);
        %imread('/usr0/home/shihenw/preprocess/patches_small/2_neck/1_1.png');
        % im = imread('peppers.png');
        % prepare oversampled input
        % input_data is Height x Width x Channel x Num
        tic;
        %input_data = {prepare_image(im)};
        input_data = {single(im_batch)};
        toc;

        % do forward pass to get scores
        % scores are now Width x Height x Channels x Num
        tic;
        s_vec = caffe('forward', input_data);
        toc;

        s_vec = s_vec{1};
        %size(scores);
        s_vec = squeeze(s_vec);
        if(j == numBatches) %last batch
            s_vec = s_vec(:,1:end-numPadding);
        end
        %s_vec = mean(s_vec,2);

        %[~,maxlabel] = max(s_vec);
        scores = [scores s_vec];%{batch*(j-1)+1:batch*j} = s_vec;
        %fprintf('%d, %d\n', r, c);
end

scores = mat2cell(scores, 2, ones(1,numImages));
scores = reshape(scores, sizeOfPatches);

maxlabel = 0;
map = cellfun(@(x) min(max(x(1),0),1), scores);

save('scores.mat', 'scores', 'map');
%figure;
%imagesc(map);
%close all;


% ------------------------------------------------------------------------
function images = prepare_image(im)
% ------------------------------------------------------------------------
d = load('ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);



function patches = generatePatches(im, psize, stride)
    %im = imread(imFilename);
    %distHtoN = sqrt((x(1) - x(2))^2 + (y(1) - y(2))^2); % from head to neck
    %scale = targetDist/distHtoN;
    %im = imresize(im, scale, 'bilinear');
    %y = y * scale;
    %x = x * scale;
    % now im,x,y are rescaled

    w = size(im, 2);
    h = size(im, 1);

    % for center points
    w_tick = psize/2+1 : stride : w-psize/2;
    h_tick = psize/2+1 : stride : h-psize/2;

    % left-up points
    w = w_tick - psize/2;
    h = h_tick - psize/2;

    [ww, hh] = meshgrid(w,h);
    tickcell = mat2cell(reshape([ww(:) hh(:)]', 2*size(ww, 1), []), 2*ones(size(h)), ones(size(w)));
    patches = cellfun(@(x) im(x(2):x(2)+psize-1, x(1):x(1)+psize-1, :), tickcell, 'UniformOutput', false);
    
    % doing contrast normalization and centering (-0.5)
    parfor row = 1:size(patches,1)
        patches(row,:) = cellfun(@(x) processOnePatch(x), patches(row,:), 'UniformOutput', false);
    end
    
function finalPatch = processOnePatch(patch)
    HSV_patch = rgb2hsv(patch);
    HSV_patch(:,:,3) = histeq(HSV_patch(:,:,3));
    finalPatch = hsv2rgb(HSV_patch);
    finalPatch = finalPatch - 0.5; %center it
