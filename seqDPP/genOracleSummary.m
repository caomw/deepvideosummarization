load('gtLabels.mat');
load('allFisher.mat');
% all sampled frames starts with 2, so replace all 1s with 2s
for v=1:size(gtLabels,1)
    for u=2:6
        if(gtLabels{v,u}(1)==1)
            gtLabels{v,u}(1)=2;
        end
    end
end
time=tic;
oracleSummary=cell(size(gtLabels,1),1);
% for each video
for v=[1:10,12,22:50]%1:size(gtLabels,1)
    % compute correlation between all frames, matched if CORR>THRESHOLD
    CORR=corr(allFisher{v});
    THRESHOLD=0.2;
    MATCHED=CORR>THRESHOLD;
    Yu=gtLabels(v,2:6)'; % user-labeled summaries
    Ystar=[]; % desired oracle summary
    F=0;
    candFrames=[2:size(allFisher{v},2)+1]; % frames to be picked from
    % greedy algorithm: synthesize oracle summary
    % y* <-- y* U argmax_i sum_u(Fscore(y*Ui,yu))
    while(true)
        Fupdate=zeros(length(candFrames),1);
        % find the frame that maximizes sum of F-scores
        for c=1:length(candFrames)
            YstarUi=sort([Ystar;candFrames(c)],'ascend');
            for u=1:5
                Fupdate(c)=Fupdate(c)+computeFscore(YstarUi,Yu{u},MATCHED);
            end
        end
        [FupdateMax,maxIdx]=max(Fupdate);
        if(FupdateMax<F)
            % break if no frames can further increase F-score
            break;
        else
            % adds frame to y*
            F=FupdateMax;
            Ystar=sort([Ystar;candFrames(maxIdx)],'ascend');
            candFrames(maxIdx)=[];
        end
    end
    oracleSummary{v}=Ystar;
    oracleFolder=['oracleSummary/',gtLabels{v,1}];
    if(~exist(oracleFolder,'dir'))
        mkdir(oracleFolder);
    end
    if(v==11||(13<=v&&v<=21)) % .flv: all frames
        sampleFolder=['allFrames227x227/',gtLabels{v,1}];
    else % .avi: 1 frame/sec
        sampleFolder=['sampledFrames227x227/',gtLabels{v,1}];
    end
    for f=1:length(oracleSummary{v})
        copyfile([sampleFolder,'/',int2str(oracleSummary{v}(f)),'.jpg'],oracleFolder);
    end
    fprintf('v=%d oracle summary done, time=%fsec\n',v,toc(time));
end
save('oracleSummary.mat','oracleSummary');

    