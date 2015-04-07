run('../VideoUtils_v1_2_4/install.m');
dirName='new_database';
videoDir=dir(dirName);
for v=4:length(videoDir)
    videoFileName=['new_database/',videoDir(v).name];
    videoName=videoFileName(length(dirName)+2:end-4);
    vp=VideoPlayer(videoFileName);
    if(~exist(['allFrames/',videoName],'dir'))
        mkdir(['allFrames/',videoName]);
    end
    % compare file extension
    fileExt=videoFileName(end-2:end);
    if(strcmp(fileExt,'avi'))
        videoFrameN=vp.NumFrames;
        videoTime=vp.TotalTime;
        videoFrameRate=videoFrameN/videoTime;
        if(~exist(['sampledFrames/',videoName],'dir'))
            mkdir(['sampledFrames/',videoName]);
        end
    elseif(strcmp(fileExt,'flv'))
        % no info, but still read
    else
        fprintf('%s: other formats!\n',videoFileName);
        continue;
    end
    % read and sample frames
    ithframe=1;
    ithsample=2;
    while(true)
        imwrite(vp.Frame,['allFrames/',videoName,'/',int2str(ithframe),'.jpg']);
        fprintf('%s: frame #%d done',videoName,ithframe);
        if(strcmp(fileExt,'avi'))
            if(ithframe==max(floor((ithsample-2)*videoFrameRate),1))
                imwrite(vp.Frame,['sampledFrames/',videoName,'/',int2str(ithsample),'.jpg']);
                fprintf(' (sampled at second %d)',ithsample);
                ithsample=ithsample+1;
            end
        end
        fprintf('\n');
        ithframe=ithframe+1;
        if(~vp.nextFrame)
            break;
        end
    end
    clear vp;
end

