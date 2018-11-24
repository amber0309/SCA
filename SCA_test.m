function [ACC, pre_labels, Zs, Zt] = SCA_test(B, A, K_s, K_t, Y_s, Y_t, eig_ratio)
%{
apply the learned transformation on test domain

INPUT:
  B           - transformation matrix
  A           - eigenvalues
  K_s         - kernel matrix of training data
  K_t         - kernel matrix of target data
  Y_s         - training label in L by 1 matrix
  Y_t         - target label in L by 1 matrix
  eig_ratio   - eigvalue ratio used for test

OUTPUT:
  ACC         - test accuracy on target domain
  pre_labels  - predicted labels of target domain data
  Zs          - projected source domain data
  Zt          - projected target domain data

Shoubo (shoubo.sub AT gmail.com)
24/11/2018

----------------------------------------------------------------------
[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.
----------------------------------------------------------------------
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

Mdl = fitcknn(Zs, Y_s);
pre_labels = predict(Mdl, Zt);
ACC = length(find(pre_labels == Y_t)) / length(pre_labels);