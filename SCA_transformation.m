function [B, A] = cpt_sca_b(P, T, D, Q, K_bar, beta, delta, epsilon)
%{
implementation of scatter component analysis [1]

INPUT:
  X           - cell of L by d matrix, each matrix corresponds to the data of a domain
  Y           - cell of L by 1 matrix, each matrix corresponds to the label of a domain
  beta, delta - trade-off parameters in Eq.(20) in [1]
  epsilon     - a small constant for numerical stability
  sigma       - kernel width

OUTPUT:
  A           - eigenvalues
  B           - transformation matrix

Shoubo (shoubo.sub AT gmail.com)
06/25/2018
----------------------------------------------------------------------

[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.
%}

%----------------------------------------------------------------------

%compute transformation B
I_0 = eye(size(K_bar, 1));
F1 = beta * P + (1 - beta) * T; % P between class scatter; T total scatter
F2 = ( delta * D + Q + K_bar + epsilon*I_0); % D domain scatter; Q within class scatter
F = F2\F1;

[B, A] = eig(F);
B = real(B);
A = real(A);
eigvalues = diag(A);
[val, idx] = sort(eigvalues, 'descend');
B = B(:, idx);
A= diag(val);