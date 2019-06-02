function [acc, pre_labels, Zs, Zt] = SCA_test(B, A, K_s, K_t, Y_s, Y_t, eig_ratio)
%{
test transformation learned by scatter component analysis [1]

INPUT:
  A           - eigenvalues
  B           - transformation matrix
  X_all       - train data in cell format, each element is a L by d matrix
  X_s         - train data in L by d matrix format
  Y_s         - train label in L by 1 matrix format
  X_t         - target domain data in L by d matrix
  Y_t         - target domain label in L by 1 matrix
  sigma       - kernel width
  eig_ratio   - eigvalue ratio used for test

OUTPUT:
  ACC         - test accuracy on target domain
  pre_labels  - predicted labels of target domain data
  Zs          - projected source domain data
  Zt          - projected target domain data

Shoubo (shoubo.sub AT gmail.com)
06/25/2018
----------------------------------------------------------------------

[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.

%}

% X_s = cat(1, X{:});
% Y_s = cat(1, Y{:});
% X_total = X_s;

% n_s = size(X_s, 1);
% n_t = size(X_t, 1);
% n_total = size(X_s, 1);

% H = eye(n_total) - ones(n_total)./n_total;
% H_t = eye(n_t) - ones(n_t)./n_t;
% H_s = eye(n_s) - ones(n_s)./n_s;

% B = real(B);
% A = real(A);

% dist = pdist2(X_total, X_t);
% dist = dist.^2;
% K_t = exp(-dist./(2 * sigma * sigma));

% K_t = H * K_t * H_t;

vals = diag(A);
ratio = [];
count = 0;
for i = 1:length(vals)
    if vals(i)<0
        break;
    end
    count = count + vals(i);
    ratio = [ratio; count];
    vals(i) = 1/sqrt(vals(i));
end
A_sqrt = diag(vals);
ratio = ratio/count;

if eig_ratio <= 1
    idx = find(ratio>eig_ratio);
    n_eigs = idx(1);
else
    n_eigs = eig_ratio;
end

Zt = K_t' * B(:, 1:n_eigs) * A_sqrt(1:n_eigs, 1:n_eigs);

% dist = pdist2(X_total, X_s);
% dist = dist.^2;
% K_s = exp(-dist./(2*sigma*sigma));
% K_s = H*K_s*H_s;

Zs = K_s' * B(:, 1:n_eigs) * A_sqrt(1:n_eigs, 1:n_eigs);

% ----- 1NN classifier
% Mdl = fitcknn(Zs, Y_s);
% pre_labels = predict(Mdl, Zt);
% acc = length(find(pre_labels == Y_t)) / length(pre_labels);
 
% ----- SVM classifier
dist_s_s = pdist2(Zs, Zs);
dist_s_s = dist_s_s.^2;
half_dist = dist_s_s-tril(dist_s_s);
half_dist = reshape(half_dist, size(dist_s_s, 1)^2, 1);
md = sqrt(median(half_dist(half_dist>0)));

t = templateSVM('KernelFunction', 'rbf', 'KernelScale', md, 'IterationLimit', 1e3);
Mdl = fitcecoc(Zs, Y_s, 'Coding', 'onevsall', 'Learners', t);
pre_labels = predict(Mdl, Zt);
acc = length(find(pre_labels==Y_t))/length(pre_labels);