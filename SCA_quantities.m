function [P, T, D, Q, K_bar] = cpt_sca_terms(K, X, Y)
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

% number of domains
n_domain = length(X);

% total number of obs from all domains
n_total = 0;

% # of obs per domain
nper_domain = [0];
count = 0;
for i =1:n_domain
    n_total = n_total + size(X{i}, 1);
    count = count + size(X{i}, 1);
    nper_domain = [nper_domain count];
end

% % obs of X from all domains
% X_total = cat(1, X{:});

% %compute kernel matrix K:
% dist = pdist2(X_total, X_total);
% dist = dist.^2;
% K = exp(-dist./( 2 * sigma * sigma));

% labels of all domains in a vector
Y_ALL = cat(1, Y{:});

idx = find(Y_ALL==0);

% check the label of the first class
if length(idx)~=0
   num_class = length(unique(Y_ALL))-1;   %The num of classes mush begin from one
else
   num_class = length(unique(Y_ALL));
end

%find the samples of class k and put them into cell k;
X_c = cell(1, num_class);
for i = 1:num_class
    X_c{i} = [];
end

% save class and domain index of all obs into two row vectors 
class_index = zeros(1, n_total);
domain_index = zeros(1, n_total);
count = 1;
num_labeled = 0;
for i = 1:n_domain
    for j = 1:size(Y{i}, 1)
        temp_c = Y{i}(j);
        class_index(count) = temp_c;
        domain_index(count) = i;
        count = count + 1;
        if temp_c~= 0
            X_c{temp_c} = [X_c{temp_c}; X{i}(j, :)];
            num_labeled = num_labeled + 1;
        end
    end
end


%compute matrix P
P = zeros(n_total, n_total);
class_idx = zeros(num_labeled, 1);
count = 1;

for i = 1:num_class
    j = 0;
    while j < size(X_c{i}, 1)
        class_idx(count) = i;
        count = count + 1;
        j = j + 1;
    end
end

P_mean = mean(K(:, class_index ~= 0), 2);

for j = 1:num_labeled
    class_id = class_idx(j);
    temp_k = mean(K(:, class_index == class_id), 2);
    P(:,j) = temp_k - P_mean;
end
P = P*P';

%compute matrix Q
Q = zeros(n_total, n_total);
for i = 1:num_class
    
    idx = find(class_index==i);
    G_j = mean(K(:, idx), 2);
    
    G_ij = K(:, idx);
    Q_i = G_ij - repmat(G_j, 1, length(idx));
    
    Q = Q+Q_i*Q_i';
end

% compute matrix in domain scatter
D = zeros(n_total, n_total);
temp = zeros(n_total, 1);
for j = 1:n_domain
    % idx = find(domain_index==j);
    temp = temp + mean(K(:, domain_index==j), 2);
end
temp = temp / n_domain;

for j = 1:n_domain
    % idx = find(domain_index==j);
    D(:,j) = mean(K(:, domain_index==j), 2) - temp;
end
D = D * (D') / n_domain;

I = ones(n_total, n_total)*(1/n_total);
K_bar = K - I*K - K*I + I*K*I;

% compute matrix in total scatter
% K_M = K - repmat(mean(K, 2), 1, n_total);
T = K_bar * K_bar / n_total;


% %compute transformation B
% I_0 = eye(n_total);
% F1 = beta*P + (1 - beta) * T; % P between class scatter; T total scatter
% F2 = ( delta * D + Q + K + epsilon*I_0); % D domain scatter; Q within class scatter
% F = F2\F1;

% [B, A] = eig(F);
% B = real(B);
% A = real(A);
% eigvalues = diag(A);
% [val, idx] = sort(eigvalues, 'descend');
% B = B(:, idx);
% A= diag(val);