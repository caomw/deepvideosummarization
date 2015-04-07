resizeH=227;
resizeW=227;
allFramesVideoDirName='allFrames';
allFramesVideoDir=dir(allFramesVideoDirName);
for v=3:length(allFramesVideoDir)
    if(~exist(['allFrames227x227/',allFramesVideoDir(v).name],'dir'))
        mkdir(['allFrames227x227/',allFramesVideoDir(v).name]);
    end
    allFramesDirName=['allFrames/',allFramesVideoDir(v).name];
    allFramesDir=dir(allFramesDirName);
    for f=3:length(allFramesDir)
        img=imread([allFramesDirName,'/',allFramesDir(f).name]);
        imgResize=imresize(img,[resizeH,resizeW]);
        imwrite(imgResize,['allFrames227x227/',allFramesVideoDir(v).name,'/',allFramesDir(f).name]);
        fprintf('allFrames: %s: resize %s done\n',allFramesVideoDir(v).name,allFramesDir(f).name);
    end
end
sampledFramesVideoDirName='sampledFrames';
sampledFramesVideoDir=dir(sampledFramesVideoDirName);
for v=3:length(sampledFramesVideoDir)
    if(~exist(['sampledFrames227x227/',sampledFramesVideoDir(v).name],'dir'))
        mkdir(['sampledFrames227x227/',sampledFramesVideoDir(v).name]);
    end
    sampledFramesDirName=['sampledFrames/',sampledFramesVideoDir(v).name];
    sampledFramesDir=dir(sampledFramesDirName);
    for f=3:length(sampledFramesDir)
        img=imread([sampledFramesDirName,'/',sampledFramesDir(f).name]);
        imgResize=imresize(img,[resizeH,resizeW]);
        imwrite(imgResize,['sampledFrames227x227/',sampledFramesVideoDir(v).name,'/',sampledFramesDir(f).name]);
        fprintf('sampledFrames: %s: resize %s done\n',sampledFramesVideoDir(v).name,sampledFramesDir(f).name);
    end
end

