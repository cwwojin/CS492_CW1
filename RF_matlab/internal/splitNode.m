function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

%visualise = 0;

% Initilise child nodes
%iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);

%feature selection
j = param.splitNum;

%axis-aligned
%features = randsample(D-1,j);   %Features to analyze
%iter = j;

%two-pixel
features = randsample(D-1,j);
combs = nchoosek(features,2);       %choose 2 features
features = [combs; flip(combs,2)];  %consider order
iter = size(features,1);



ig_best = -inf; % Initialise best information gain
idx_best = [];

for n = 1:iter
    
    % Split function - Modify here and try other types of split function
    
    %Axis-aligned
%     dim = features(j);
%     for i = 1+1:N-1
%         %test all data samples for best threshold
%         t = (data(i-1,dim)+data(i+1,dim))/2;
%         idx_ = data(:,dim) < t;
% 
%         ig = getIG(data,idx_); % Calculate information gain
%         if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
%             [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
%         end
%     end

    %2-pixel test
    dim = features(j,:);
    for i = 1+1:N-1
        %test all data samples for best threshold
        t = ((data(i-1,dim(1))-data(i-1,dim(2))) + (data(i+1,dim(1))-data(i+1,dim(2)))) / 2;
        idx_ = data(:,dim(1))-data(:,dim(2)) < t;

        ig = getIG(data,idx_); % Calculate information gain
        if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
            [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
        end
    end

end
nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);
if node.dim == -1
    node.dim
    node.t
    node.idx
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end