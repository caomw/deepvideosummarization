userSummDirName='newUserSummary';
userSummDir=dir(userSummDirName);
gtLabels=cell(length(userSummDir)-2,6);
gtLabels(:,1)=arrayfun(@(X) X.name,userSummDir(3:end,1),'uniformoutput',false);
for v=3:length(userSummDir)
    userDirName=[userSummDirName,'/',userSummDir(v).name];
    userDir=dir(userDirName);
    if(strcmp(userDir(3).name,'Thumbs.db'))
        userDir(3)=[];
    end
    for u=3:length(userDir)
        gtDirName=[userDirName,'/',userDir(u).name];
        gtDir=dir(gtDirName);
        if(strcmp(gtDir(3).name,'Thumbs.db'))
            gtDir(3)=[];
        end
        gtLabels{v-2,u-1}=sort(cell2mat(arrayfun(@(X) str2double(X.name(6:end-4)),gtDir(3:end),'uniformoutput',false)),'ascend');
    end
end
save('gtLabels.mat','gtLabels');
