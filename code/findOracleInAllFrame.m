%function findOracleInAllFrame()
clear all;
load ../data/deepvideosummarization-master/oracleSummary.mat;
load ../data/deepvideosummarization-master/gtLabels.mat;

videoFolder = '../data/allFrames227x227/';
sampledVideoFolder = '../data/sampledFrames227x227/';
oracleIndexInAll = cell(size(gtLabels,1), 5); % what we really want

parfor i = 1:size(gtLabels,1) % for every video
    for u = 1:5
        fprintf('%d %d\n', i, u);

        videoName = gtLabels{i,1};
        videoSubFolder = [videoFolder videoName '/'];
        sampledSubFolder = [sampledVideoFolder videoName '/'];
        allFramesInVideo = dir([videoSubFolder '*.jpg']);
        sampledFramesVideo = dir([sampledSubFolder '*.jpg']);

        oracle = gtLabels{i,u+1};%oracleSummary{i,1};
        if(length(oracleSummary{i,1})==0) 
            continue;
        end
        % load gt (oracle) from sampled frame
        currentOracleIndex = 1;
        sampledIndex = oracle(currentOracleIndex);
        if(sampledIndex == 1)
            sampledIndex = 2;
        end
        currentOracleFrame = imread(sprintf('%s/%d.jpg', sampledSubFolder, sampledIndex));
        %fprintf('finding matching frame for sampled %s...', sprintf('%s/%d.jpg', sampledSubFolder, sampledIndex));
        
        diff = zeros(1,length(allFramesInVideo));
        % go through all frame to find oracle number
        terminate_flag = 0;
        
        for j = 1:length(allFramesInVideo)
            if(terminate_flag)
                continue;
            end
            currentAllFrame = imread(sprintf('%s/%d.jpg', videoSubFolder, j));
            diff(j) = sum(sum(sum(currentOracleFrame - currentAllFrame)));
            %fprintf('%d\n', diff);

            if(diff(j) == 0)
                %fprintf('%d\n', j);
                %imshowpair(currentAllFrame, currentOracleFrame);
                oracleIndexInAll{i,u} = [oracleIndexInAll{i,u} [oracle(currentOracleIndex); j]];
                currentOracleIndex = currentOracleIndex + 1;
                if(currentOracleIndex > length(oracle))
                    terminate_flag = 1;
                    continue; % like a break;
                end
                sampledIndex = oracle(currentOracleIndex);
                currentOracleFrame = imread(sprintf('%s/%d.jpg', sampledSubFolder, sampledIndex));
                %fprintf('finding matching frame for sampled %s...', sprintf('%s/%d.jpg', sampledSubFolder, sampledIndex));
                %drawnow;
                %pause;
            end
        end
    end
end

save('oracleIndexInAll', '../data/oracleIndexInAll.mat');