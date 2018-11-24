function [B, A] = SCA_transformation(P, T, D, Q, K_bar, beta, delta, epsilon)
%{
compute the domain-invariant transformation of scatter component analysis [1]

INPUT:
    P                   - between-class scatter
    T                   - total scatter
    D                   - domain scatter
    Q                   - within-class scatter
    K_bar               - the centered kernel matrix (K)
    beta, aph, gamma    - trade-off parameters in Eq.(20)
    epsilon             - coefficient of the identity matrix (footnote in page 5)

OUTPUT:
    B                   - matrix of projection
    A                   - corresponding eigenvalues

Shoubo (shoubo.sub AT gmail.com)
24/11/2018

----------------------------------------------------------------------
[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.
----------------------------------------------------------------------
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