load('../syn_data/f12t3_dm2_prpt4_X.mat');
load('../syn_data/f12t3_dm2_prpt4_Y.mat');

% ----- parameters
% target / all / source domains
targ_dm = [4];
vldt_dm = [3];
src_dm = [1 2];

eig_ratio = 2;
epsilon = 0.00001;

% ----- prepare data
X_train_cell = cell(1, length(src_dm));
Y_train_cell = cell(1, length(src_dm));
for src_idx = 1:length(src_dm)
    % data of domain 1
    dm_id = src_dm(src_idx);
    X_train_cell{1, src_idx} = [X{3*(dm_id-1) + 1};X{3*(dm_id-1) + 2};X{3*(dm_id-1) + 3}];
    Y_train_cell{1, src_idx} = [Y{3*(dm_id-1) + 1};Y{3*(dm_id-1) + 2};Y{3*(dm_id-1) + 3}];
end

%data of domain 3 (as validation set)
dm_id = vldt_dm(1);
X_vldt_knn = [X{3*(dm_id-1) + 1};X{3*(dm_id-1) + 2};X{3*(dm_id-1) + 3}];
Y_vldt_knn = [Y{3*(dm_id-1) + 1};Y{3*(dm_id-1) + 2};Y{3*(dm_id-1) + 3}];

%data of domain 4 (as target domain)
dm_id = targ_dm(1);
X_test_knn = [X{3*(dm_id-1) + 1};X{3*(dm_id-1) + 2};X{3*(dm_id-1) + 3}];
Y_test_knn = [Y{3*(dm_id-1) + 1};Y{3*(dm_id-1) + 2};Y{3*(dm_id-1) + 3}];

Train_data = [X_train_cell{1, 1};X_train_cell{1, 2}];
Train_label = [Y_train_cell{1, 1};Y_train_cell{1, 2}];

fprintf('Data preparation done. Conducting SCA ... \n');

% ----- distance matrix computation 
% ----- ----- source domain
X_total = cat(1, X_train_cell{:});
dist = pdist2(X_total, X_total);
dist = dist.^2;

X_s = X_total;
Y_s = cat(1, Y_train_cell{:});

n_s = size(X_s, 1);
n_t = size(X_test_knn, 1);
n_v = size(X_vldt_knn, 1);
n_total = size(X_total, 1);

H_total = eye(n_total) - ones(n_total)./n_total;
H_t = eye(n_t) - ones(n_t)./n_t;
H_s = eye(n_s) - ones(n_s)./n_s;
H_v = eye(n_v) - ones(n_v)./n_v;

% ----- ----- validation and test domains
dist_t = pdist2(X_total, X_test_knn);
dist_t = dist_t.^2;
dist_v = pdist2(X_total, X_vldt_knn);
dist_v = dist_v.^2;
% ----- ----- ----- -----

All_acc = zeros(5, 10, 7);

sigma = power( median(dist(:) ), 2);
for k = 1
    K = exp(-dist./(2 * sigma * sigma));
    K_s = H_total * K * H_s;
    K_v = exp(-dist_v./(2 * sigma * sigma));
    K_v = H_total * K_v * H_v;

    [P, T, D, Q, K_bar] = SCA_quantities(K, X_train_cell, Y_train_cell);

    for i = 1:5
        for j = 1:10
            beta = 0.1+(i-1)*0.2;
            delta = power(10, j-1);
            [B, A] = SCA_transformation(P, T, D, Q, K_bar, beta, delta, epsilon);
            
            [ACC, ~, ~, ~] = SCA_test(B, A, K_s, K_v, Y_s, Y_vldt_knn, eig_ratio);

            All_acc(i,j,k) = ACC;
            fprintf('i: %d,j: %d,k: %d, acc: %d \n', i, j, k, ACC);
        end
    end
end

fprintf('Conducting test ... \n');

ACC_tr_best = max(All_acc(:));
ind = find( All_acc==max(All_acc(:)) );
[m, n, x] = size(All_acc);
[best_i, best_j, best_k] = ind2sub([m n x], ind(1));

best_beta = 0.1+(best_i-1)*0.2;
best_delta = power(10, best_j-1);
best_sigma = power( median(dist(:) ), 2);

K = exp( -dist./( 2* best_sigma * best_sigma ) );
K_s = H_total * K * H_s;
K_t = exp(-dist_t./(2 * best_sigma * best_sigma));
K_t = H_total * K_t * H_t;

[P, T, D, Q, K_bar] = SCA_quantities(K, X_train_cell, Y_train_cell);
[B, A] = SCA_transformation(P, T, D, Q, K_bar, best_beta, best_delta, epsilon);

[ACC_final, ~, Zs, Zt] = SCA_test(B, A, K_s, K_t, Y_s, Y_test_knn, eig_ratio);

Z = [Zs; Zt];
y_z = [ Y_s; Y_test_knn];

fprintf('Final test accuracy: %f\n', ACC_final);