run('../vlfeat-0.9.20/toolbox/vl_setup');
load('gtLabels.mat');
numClusters=32; % to make Fisher dimension=8192
allFisher=cell(size(gtLabels,1),1);
time=tic;
% for each video
for v=1:size(gtLabels,1)
    % generate GMM for Fisher: use all frames of a video
    trainDir=dir(['allFrames227x227/',gtLabels{v,1}]);
    Dall=[];
    for f=4:length(trainDir)
        % extract SIFT of each frame
        I=imread(['allFrames227x227/',gtLabels{v,1},'/',trainDir(f).name]);
        Igs=single(rgb2gray(I));
        [~,D]=vl_sift(Igs);
        Dall=[Dall,D];
        fprintf('v=%d generate GMM: f=%d/%d done, time=%fsec\n',v,f,length(trainDir),toc(time));
    end
    % prevent out-of-memory in Fisher: randomly select at most 4M samples
    if(size(Dall,2)>4000000)
        randpermIdx=randperm(size(Dall,2),4000000);
        [mu,sigma,prior]=vl_gmm(single(Dall(:,randpermIdx)),numClusters);
    else
        [mu,sigma,prior]=vl_gmm(single(Dall),numClusters);
    end
    fprintf('v=%d generate GMM done, time=%fsec\n',v,toc(time));
    % compute Fisher of all frames: sampled frames
    if(v==11||(13<=v&&v<=21)) % .flv: all frames
        sampleFolder='allFrames227x227/';
    else % .avi: 1 frame/sec
        sampleFolder='sampledFrames227x227/';
    end
    sampleDir=dir([sampleFolder,gtLabels{v,1}]);
    allFisher{v}=zeros(128*numClusters*2,length(sampleDir)-2);
    for f=3:length(sampleDir)
        % extract SIFT of each frame
        I=imread([sampleFolder,gtLabels{v,1},'/',sampleDir(f).name]);
        Igs=single(rgb2gray(I));
        [~,D]=vl_sift(Igs);
        % compute Fisher vector
        E=vl_fisher(single(D),mu,sigma,prior);
        allFisher{v}(:,f-2)=E;
        fprintf('v=%d compute Fisher: f=%d/%d done, time=%fsec\n',v,f,length(sampleDir),toc(time));
    end
end
save('allFisher.mat','allFisher');
