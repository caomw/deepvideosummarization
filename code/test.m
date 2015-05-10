function test(videoID, K)

addpath /usr0/home/shihenw/caffe_701/caffe/matlab/caffe;

caffe('reset');
modelName = '../exp_caffe/exp3_real/video_iter_30941.caffemodel';
deployFileName = '../exp_caffe/exp3_real/deploy.prototxt';
matcaffe_init(1, deployFileName, modelName);

load ../data/deepvideosummarization-master/oracleSummary.mat;
load ../data/deepvideosummarization-master/gtLabels.mat;
load ../data/oracleIndexInAll.mat;

videoName = gtLabels{videoID,1};
disp(videoName);
videoFolder = '../data/allFrames227x227/';
sampledVideoFolder = '../data/sampledFrames227x227/';
oracleIndexInAll = cell(size(gtLabels,1), 5);

videoSubFolder = [videoFolder videoName '/'];
sampledSubFolder = [sampledVideoFolder videoName '/'];
allFramesInVideo = dir([videoSubFolder '*.jpg']);
sampledFramesVideo = dir([sampledSubFolder '*.jpg']);

numSampledFrame = length(sampledFramesVideo);
features = zeros(numSampledFrame, 1000);

%DATA = h5read('../data/h5file_test/batch_v50_1_0001.h5', '/data');
% DATA = permute(DATA, [2 1 3 4]);
% DATA = DATA(:,:,[3 2 1],:);
%LABEL = h5read('../data/h5file_test/batch_v50_1_0001.h5', '/label');

for i = 1:numSampledFrame
    fprintf('%d ', i);
  
    frame = imread(sprintf('%s/%d.jpg', sampledSubFolder, i+1));
    %imshow(frame);
    input_data = {preprocess(frame)};
    %input_data = {DATA; LABEL};
    scores = caffe('forward', input_data);
    scores = scores{1};
    features(i,:) = scores;
end

fprintf('\n');

%save(sprintf('../deepFeature/%02d', videoID), 'features');
[IDX, C, SUMD, D] = kmeans(features, K);
[Y,I] = min(D);
disp(sort(I));
for i = I
    frame = imread(sprintf('%s/%d.jpg', sampledSubFolder, i+1));
    imshow(frame);
    pause;
end


function im_p = preprocess(im)
    im_p = single(im)/255 - 0.5;