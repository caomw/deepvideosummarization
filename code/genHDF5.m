function genHDF5()

load ../data/oracleIndexInAll.mat;
load ../data/deepvideosummarization-master/gtLabels.mat;

batch_size = 128;
image_size = 227;

%count = 0;
for i = size(gtLabels,1)-4 : size(gtLabels,1) % for every video
    if(isempty(oracleIndexInAll{i,1}))
        continue;
    end
    videoName = gtLabels{i,1};
    videoFolder = '../data/allFrames227x227/';
    videoSubFolder = [videoFolder videoName '/'];
        
    for u = 1:5
        fprintf('%d movie, %d user\n', i, u);
        j_gt = oracleIndexInAll{i,u}(2,:); % keyframes
        
        %get total number of frame
        frames = dir([videoSubFolder '/*.jpg']);
        numFrames = length(frames);
        remainFrame = numFrames - length(j_gt);
        remainSlot = batch_size - length(j_gt); % for each h5 file
 
        i_other = setdiff(1:numFrames, j_gt);
        i_other = i_other(1:3:end); % sample frame
        remainFrame = length(i_other);
        i_other = i_other(randperm(remainFrame));
        assert(length(i_other) == remainFrame);
        
        numFiles = ceil(remainFrame/remainSlot);
        
        for f = 1:numFiles
            %count = count + 1;
            h5name = sprintf('../data/h5file_test/batch_v%02d_%d_%04d.h5', i, u, f);
            
            if(exist(h5name, 'file'))
                continue;
            end
            
            data = zeros(image_size, image_size, 3, batch_size);
            label = zeros(1, batch_size);
            
            for j = 1:length(j_gt)
                im = imread(sprintf('%s/%d.jpg', videoSubFolder, j_gt(j)));
                data(:,:,:,j) = preprocess(im);
                label(:,j) = length(j_gt);
            end
            remainSection = remainSlot*(f-1)+1 : min(remainSlot*f, remainFrame); 
            for ii = 1:length(remainSection)
                fprintf('%d\t', remainSection(ii));
                im = imread(sprintf('%s/%d.jpg', videoSubFolder, i_other(remainSection(ii))));
                %fprintf('index: %d, \n', ii+length(j_gt));
                data(:,:,:,ii+length(j_gt)) = preprocess(im);
                label(:,ii+length(j_gt)) = length(j_gt);
            end
            % save hdf5
            
            assert(size(data, 4) == batch_size);
            h5create(h5name, '/data', size(data), 'Datatype', 'single');
            h5create(h5name, '/label', size(label), 'Datatype', 'single');
            h5write(h5name, '/data', single(data));
            h5write(h5name, '/label', single(label));
            %fprintf('%d ', count);
        end
    end
end

function im_p = preprocess(im)
    im_p = single(im)/255 - 0.5;