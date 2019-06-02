function [acc, pre_labels, Zs, Zt] = SCA_test(B, A, K_s, K_t, Y_s, Y_t, eig_ratio)
%{
Apply the transformation on the target domain instances

INPUT:
  B           - transformation matrix
  A           - eigenvalues
  K_s         - (n_s*n_s) kernel matrix of source instances
  K_t         - (n_s*n_t) kernel matrix between source and target instances 
  Y_s         - (n*1) matrix of class labels of source instances 
  Y_t         - (n_t*1) matrix of class labels of target instances
  eig_ratio   - dimension of the transformed space

OUTPUT:
  acc         - test accuracy on target instances
  pre_labels  - predicted labels of target instances
  Zs          - projected source domain instances
  Zt          - projected target domain instances

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-02

Reference
[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.
%}

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

Zs = K_s' * B(:, 1:n_eigs) * A_sqrt(1:n_eigs, 1:n_eigs);

% ----- 1NN classifier
Mdl = fitcknn(Zs, Y_s);
pre_labels = predict(Mdl, Zt);
acc = length(find(pre_labels == Y_t)) / length(pre_labels);
 
% ----- SVM classifier
% dist_s_s = pdist2(Zs, Zs);
% dist_s_s = dist_s_s.^2;
% half_dist = dist_s_s-tril(dist_s_s);
% half_dist = reshape(half_dist, size(dist_s_s, 1)^2, 1);
% md = sqrt(median(half_dist(half_dist>0)));

% t = templateSVM('KernelFunction', 'rbf', 'KernelScale', md, 'IterationLimit', 1e3);
% Mdl = fitcecoc(Zs, Y_s, 'Coding', 'onevsall', 'Learners', t);
% pre_labels = predict(Mdl, Zt);
% acc = length(find(pre_labels==Y_t))/length(pre_labels);