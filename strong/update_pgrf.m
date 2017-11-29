function [rf, result] = update_pgrf(input, mask, rf, N_TREE)

Beta = 0.15;
G_para = 0.5;

maxDepth = 10;
maxDepth2 = 5;
maxNNode = 2^(maxDepth);
AMBIGUITY_THRESH = 0.40;

TOL = 10;

origin_bgProb = mask(:);

if(size(input,3) > 1)
    input = cat(3, input, rgb2lab(input) / 255 + 0.5);
end
feature = reshape(input, size(input,1)*size(input,2), size(input,3));

% training initial forest
inputFeature = cell(N_TREE,1);
inputBgProb = cell(N_TREE,1);

for i = 1:N_TREE
    idx = randsample(size(feature,1),round(0.8*(size(feature,1))));
    inputFeature{i,1} =  feature(idx,:);
    inputBgProb{i,1} = origin_bgProb(idx,:);
end


%training (50ms)
trees = rf.mainTree(1:N_TREE);

% predict (70ms)
prob = zeros(size(feature,1),N_TREE);
d = zeros(maxNNode,N_TREE);
for i = 1:N_TREE
    [yfit, c] = trees{i,1}(feature);
    leafIdx = unique(c);
    counts1 = histc(c, leafIdx);
    counts2 = histc(c(origin_bgProb==1), leafIdx);
    d(leafIdx,i) = counts2 ./ counts1;
    prob(:,i) = str2num(cell2mat(yfit(:)));
end
prob = mean(prob,2);

bgProb = prob;

bgDist = 1 - reshape(bgProb,size(mask));
fgDist = 1 - bgDist;


%%% Get the image gradient
gradH = input(:,2:end,:) - input(:,1:end-1,:);
gradV = input(2:end,:,:) - input(1:end-1,:,:);

gradH = sum(gradH.^2, 3);
gradV = sum(gradV.^2, 3);

hC = exp(-Beta.*gradH./mean(gradH(:)));
vC = exp(-Beta.*gradV./mean(gradV(:)));

%%% These matrices will evantually use as inputs to Bagon's code
hC = [hC zeros(size(hC,1),1)];
vC = [vC ;zeros(1, size(vC,2))];
sc = [0 G_para;G_para 0];

fgDist(mask(:)==1) = max(max(fgDist));
bgDist(mask(:)==1) = min(min(bgDist));

dc = cat(3, exp(bgDist), log(fgDist+1));
dc = cat(3, (bgDist), (fgDist));
graphHandle = GraphCut('open', dc , sc, vC, hC);
graphHandle = GraphCut('set', graphHandle, int32(mask == 0));
[graphHandle currLabel] = GraphCut('expand', graphHandle,1000);
currLabel = 1 - currLabel;
GraphCut('close', graphHandle);

bgProb = double(currLabel);
prevLabel = currLabel;

for ii = 1:5
    
    ratio = sum(currLabel(:)) / length(currLabel(:));
    
    label = bgProb(:);
        
    
    % predict
    prob = zeros(size(feature,1),N_TREE);
    c = zeros(size(feature,1),N_TREE);
    d = zeros(maxNNode,N_TREE);
    for i = 1:N_TREE
        [yfit, c2] = eval(trees{i,1},feature);
        leafIdx = unique(c2);
        counts1 = histc(c2, leafIdx);
        counts2 = histc(c2(label==1), leafIdx);
        if(size(counts1,1) ~= size(counts2,1))
            counts2 = counts2';
        end
        d(leafIdx,i) = counts2 ./ counts1;
        
        prob(:,i) = str2num(cell2mat(yfit(:)));
        c(:,i) = c2;
    end
    
    
    leaf2partialTree = zeros(size(d));
    num_partialTree = sum(sum(d > AMBIGUITY_THRESH & d < 1-AMBIGUITY_THRESH));
    
    if(num_partialTree < 1)
        prob = mean(prob,2);
        bgProb = prob;

        bgDist = 1 - reshape(bgProb,size(mask));
            
        bgDist(mask(:)==1) = min(min(bgDist));
        
        d3 = [];
        partialTree = [];
        break;
    end
    
    partialTree = cell(num_partialTree,1);
    k = 1;
    d3 = cell(num_partialTree, 1);
    for j = 1:N_TREE
        idx = find(d(:,j) > AMBIGUITY_THRESH & d(:,j) < 1-AMBIGUITY_THRESH);
        for i = 1:length(idx)
            featureIdx = find(c(:,j)==idx(i));
            
            if(~isempty(featureIdx))
                
                partialFeature = feature(featureIdx,:);
                
                partialTree{k,1} = classregtree(partialFeature, bgProb(featureIdx),'maxdepth',maxDepth2,'method','classification');
                
                if(partialTree{k,1}.numnodes > 1)
                    
                    d3{k,1} = classprob(partialTree{k,1});
                    [yfit, c3] = eval(partialTree{k,1}, partialFeature);
                    
                    prob(featureIdx,j) = d3{k,1}(c3(:),2);
                    
                    leaf2partialTree(idx(i),j) = k;
                    k = k + 1;
                    
                end
                
            end
            
        end
    end
    
    prob = mean(prob,2);
    bgProb = prob;
    
    bgDist = 1 - reshape(bgProb,size(mask));
    fgDist = 1 - bgDist;
    
    %%% Get the image gradient
    gradH = input(:,2:end,:) - input(:,1:end-1,:);
    gradV = input(2:end,:,:) - input(1:end-1,:,:);
    
    gradH = sum(gradH.^2, 3);
    gradV = sum(gradV.^2, 3);
    
    hC2 = exp(-Beta.*gradH./mean(gradH(:)));
    vC2 = exp(-Beta.*gradV./mean(gradV(:)));
    
    %%% These matrices will evantually use as inputs to Bagon's code
    hC = [hC2 zeros(size(hC2,1),1)];
    vC = [vC2 ;zeros(1, size(vC2,2))];
    sc = [0 G_para;G_para 0];
    
    fgDist(mask(:)==1) = max(max(fgDist));
    bgDist(mask(:)==1) = min(min(bgDist));
    
    dc = cat(3, (bgDist), (fgDist));
    graphHandle = GraphCut('open', dc , sc, vC, hC);
    graphHandle = GraphCut('set', graphHandle, prevLabel);
    [graphHandle, currLabel] = GraphCut('expand', graphHandle, 1000);
    currLabel = 1 - currLabel;
    GraphCut('close', graphHandle);
        
    bgProb = double(currLabel);
    
    if(sum(abs(prevLabel - currLabel)) < TOL)
        break;
    else
        prevLabel = currLabel;
    end
    
end

result = bgDist;

rf.mainTree = trees;
rf.leaf2partialTree = leaf2partialTree;
if(size(partialTree,1)<1)
    rf.partialTree = [];
else
    rf.partialTree = partialTree;
end
rf.mainProb = d;
rf.subProb = d3;