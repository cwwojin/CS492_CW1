function [coeff,latent,mu,A,V,s] = pca_fast(X)
%PCA_FAST - computationally efficient eigenfaces
mu = mean(X,1); n = size(X,1);
A = X - repmat(mu,size(X,1),1);
s = (A * A')./n;
[V,D] = eig(s);
[latent, ind] = sort(diag(D),'descend');
V = V(:,ind);
v = A' * V;
norm = vecnorm(v);
coeff = v ./ repmat(norm,size(v,1),1);
end

