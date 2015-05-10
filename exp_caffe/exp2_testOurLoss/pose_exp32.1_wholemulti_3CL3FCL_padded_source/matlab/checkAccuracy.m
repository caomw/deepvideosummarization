function checkAccuracy(model_file)
%clear all;
addpath('/usr0/home/shihenw/');
matcaffe_init(1, '../pose_deploy.prototxt', model_file);

paths = textread('../filelist_train.txt', '%s');

totalTrain = zeros(length(paths), 1);
correct = zeros(length(paths), 1); % stats

% try
%     %fprintf('Closing any pools...\n');
%     matlabpool close; 
% catch ME
%     disp(ME.message);
% end
% 
% matlabpool('local', 10);
batch = 16;
p = 9;
%C_tr = zeros(np+1,np+1);%cell(1,length(paths));

loss = 0;
totalItem = 0;

for i = 1:length(paths)
    fprintf('chechAccuracy_tr: %d of %d\n', i, length(paths));
    data = h5read(paths{i}, '/data');
    label = h5read(paths{i}, '/label');
    
    numImages = size(data, 4);
    %imageCell = mat2cell(data, size(data,1), size(data,2), size(data,3), ones(1,numImages));
    %imageCell = squeeze(imageCell);
    
    %s_vec = caffe('forward', {data});
    numBatches = ceil(numImages/batch);
    numPadding = batch * numBatches - numImages;
    zeroPad = zeros(size(data,1), size(data,2), size(data,3), numPadding);
    data_pad = cat(4, data, zeroPad);
    
    %C{i} = zeros(2,2);
    
    %getScores_batch(128);
    for j = 1:numBatches
        %fprintf('batch %d ', j);
        im_batch = data_pad(:,:,:,batch*(j-1)+1:batch*j);
        s_vec = caffe('forward', {im_batch});
        
        if(j == numBatches) %last batch
            s_vec{1} = s_vec{1}(:,:,:,1:end-numPadding);
        end
       
        dim = size(s_vec{1});
        score = reshape(s_vec{1}, [prod(dim(1:3)) dim(4)]); % score
        groundtruth = label(:,batch*(j-1)+1 : min(batch*j,numImages));
       	error_vec = score - groundtruth;
        loss = loss + sum(sum(error_vec.^2, 1));
        totalItem = totalItem + size(error_vec, 2);
        %C_tr = C_tr + confusionmat(I1, I2, 'order', 1:np+1);
%         if (I1==I2)
%             correct(i) = correct(i) + 1;
%         end
    end
    %fprintf('file number %d of %d\n', i, length(paths));
    %totalTrain(i) = numImages;
end
loss = loss / totalItem;
fprintf('train loss: %f\n', loss);

% ----- test data now -----
totalTest = zeros(length(paths), 1);
paths = textread('../filelist_test.txt', '%s');
loss = 0;
totalItem = 0;

for i = 1:length(paths)
    fprintf('chechAccuracy_te: %d of %d\n', i, length(paths));
    data = h5read(paths{i}, '/data');
    label = h5read(paths{i}, '/label');
    
    numImages = size(data, 4);
    %imageCell = mat2cell(data, size(data,1), size(data,2), size(data,3), ones(1,numImages));
    %imageCell = squeeze(imageCell);
    
    %s_vec = caffe('forward', {data});
    numBatches = ceil(numImages/batch);
    numPadding = batch * numBatches - numImages;
    zeroPad = zeros(size(data,1), size(data,2), size(data,3), numPadding);
    data_pad = cat(4, data, zeroPad);
    
    %s_vec = caffe('forward', {data});
    for j = 1:numBatches
        %fprintf('batch %d ', j);
        im_batch = data_pad(:,:,:,batch*(j-1)+1:batch*j);
        s_vec = caffe('forward', {im_batch});
        
        if(j == numBatches) %last batch
            s_vec{1} = s_vec{1}(:,:,:,1:end-numPadding);
        end
       
        dim = size(s_vec{1});
        score = reshape(s_vec{1}, [prod(dim(1:3)) dim(4)]); % score
        groundtruth = label(:,batch*(j-1)+1 : min(batch*j,numImages));
       	error_vec = score - groundtruth;
        loss = loss + sum(sum(error_vec.^2, 1));
        totalItem = totalItem + size(error_vec, 2);
        %C_tr = C_tr + confusionmat(I1, I2, 'order', 1:np+1);
%         if (I1==I2)
%             correct(i) = correct(i) + 1;
%         end
    end
end
loss = loss / totalItem;
fprintf('test loss: %f\n', loss);
%ssave('confusionMat.mat', 'C_tr', 'C_te');

% -----------------------------------------------------------------------
