function Fab=computeFscore(A,B,MATCHED)

matchMtrx=MATCHED(A-1,B-1); % (i,j)=1 if A(i) and B(j) are matched
% find #matches frames
% dynamic programming: find the longest diagonal path of 1s
[H,W]=size(matchMtrx);
pairMtrx=zeros(H,W);
pairMtrx(:,1)=matchMtrx(:,1);
pairMtrx(1,:)=matchMtrx(1,:);
for h=2:H
    for w=2:W
        if(matchMtrx(h,w)==1)
            maxCand=[pairMtrx(h-1,w-1)];
            if(matchMtrx(h-1,w)==0)
                maxCand=[maxCand,pairMtrx(h-1,w)+1];
            end
            if(matchMtrx(h,w-1)==0)
                maxCand=[maxCand,pairMtrx(h,w-1)+1];
            end
            pairMtrx(h,w)=max(maxCand);
        elseif(matchMtrx(h,w)==0)
            pairMtrx(h,w)=max([pairMtrx(h-1,w),pairMtrx(h,w-1),pairMtrx(h-1,w-1)]);
        end
    end
end
numPairs=max(pairMtrx(:));
% compute P,R,F
Pab=numPairs/length(A);
Rab=numPairs/length(B);
Fab=2*Pab*Rab/(Pab+Rab);

