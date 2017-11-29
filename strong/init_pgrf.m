function [rf, result] = init_pgrf(input, mask, N_TREE, mainTree)

maxDepth = 10;
maxNNode = 2^(maxDepth);
AMBIGUITY_THRESH = 0.40;

origin_bgProb = mask(:);

feature = reshape(input, size(input,1)*size(input,2), size(input,3));


% training initial forest
inputFeature = cell(N_TREE,1);
inputBgProb = cell(N_TREE,1);

for i = 1:N_TREE
    idx = 1:size(feature,1);
    inputFeature{i,1} =  feature(idx,:);
    inputBgProb{i,1} = origin_bgProb(idx,:);
end

%training (50ms)
trees = cell(N_TREE,1);
if(isempty(mainTree))
    for i = 1:N_TREE
        params.M = 1;
        params.N1 = size(inputFeature{i,1},1);
        params.F1 = size(inputFeature{i,1},2);

        trees{i,1} = forestTrain(inputFeature{i,1}, inputBgProb{i,1}+1,params);

    end
    
    prob = zeros(size(feature,1),N_TREE);
    d = zeros(maxNNode,N_TREE);
    for i = 1:N_TREE
        c = forestInds(single(feature),trees{i,1}.thrs,trees{i,1}.fids,trees{i,1}.child,1);
        leafIdx = unique(c);
        counts1 = histc(c, leafIdx);
        counts2 = histc(c(origin_bgProb==1), leafIdx);
        d(leafIdx,i) = counts2 ./ counts1;
        prob(:,i) = d(c,i);
    end
    prob = mean(prob,2);
    
    leaf2partialTree = zeros(size(d));
    d3 = [];
    partialTree = [];
    
else
    
    for i = 1:N_TREE
        trees{i,1} = mainTree;
    end
    prob = zeros(size(feature,1),N_TREE);
    d = zeros(maxNNode,N_TREE);
    for i = 1:N_TREE        
        c = forestInds(single(feature),trees{i,1}.thrs,trees{i,1}.fids,trees{i,1}.child,1);
        leafIdx = unique(c);
        counts1 = histc(c, leafIdx);
        counts2 = histc(c(origin_bgProb==1), leafIdx);
        d(leafIdx,i) = counts2 ./ counts1;
        prob(:,i) = d(c,i);
    end
        
    leaf2partialTree = zeros(size(d));
    num_partialTree = sum(sum(d > AMBIGUITY_THRESH & d < 1-AMBIGUITY_THRESH));
    
    if(num_partialTree < 1)
        prob = mean(prob,2);
        
        d3 = [];
        partialTree = [];
        
    else
    
        partialTree = cell(num_partialTree,1);
        k = 1;
        d3 = cell(num_partialTree, 1);
        for j = 1:N_TREE
            idx = find(d(:,j) > AMBIGUITY_THRESH & d(:,j) < 1-AMBIGUITY_THRESH);
            for i = 1:length(idx)
                featureIdx = find(c(:,j)==idx(i));

                if(~isempty(featureIdx))

                    partialFeature = feature(featureIdx,:);
                    
                    partialLabels = origin_bgProb(featureIdx);
                    params.M = 1;
                    params.N1 = size(partialFeature,1);
                    params.F1 = size(partialFeature,2);

                    partialTree{k,1} = forestTrain(partialFeature, partialLabels+1 , params);        

                    if(partialTree{k,1}.fids > 1)
                        d3{k,1} = partialTree{k,1}.distr(:,1);    
                        [hs, ps] = forestApply(single(partialFeature), partialTree{k,1});
                        prob(featureIdx,j) = ps(:,2);

                        leaf2partialTree(idx(i),j) = k;
                        k = k + 1;

                    end

                end

            end
        end
    end
    
    prob = mean(prob,2);
    
end

rf.mainTree = trees;
rf.leaf2partialTree = leaf2partialTree;
if(size(partialTree,1)<1)
    rf.partialTree = [];
else
    rf.partialTree = partialTree;
end
rf.mainProb = d;
rf.subProb = d3;

bgDist = 1 - reshape(prob,[size(input,1),size(input,2)]);
result = bgDist;
