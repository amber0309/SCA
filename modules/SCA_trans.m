function [B, A] = SCA_trans(P, T, D, Q, K_bar, beta, delta, epsilon)
%{
Compute the transformation in scatter component analysis [1]

INPUT:
  P           - matrix induced by between-class scatter (Eq.(13) in [1])
  T           - matrix induced by total scatter
  D           - matrix induced by domain scatter 
  Q           - matrix induced by within-class scatter (Eq.(14) in [1])
  K_bar       - centered kernel matrix K
  beta, delta - trade-off parameters in Eq.(20) in [1]
  epsilon     - a small constant for numerical stability

OUTPUT:
  B           - transformation matrix
  A           - corresponding eigenvalues

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-02

Reference
[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.
%}


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