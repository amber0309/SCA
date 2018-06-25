clear all;

% ----- data preparation
load('dataX.mat');
load('dataY.mat');

% data of domain 1
X_all{2}= cat(1, X{1:3});
Y_all{2} = cat(1, Y{1:3});

%data of domain 2
X_all{1} = cat(1, X{4:6});
Y_all{1} = cat(1, Y{4:6});

%data of domain 3 (as validation set)
X_t1 = cat(1, X{7:9});
Y_t1 = cat(1, Y{7:9});

%data of domain 4 (as target domain)
X_t2 = cat(1, X{10:12});
Y_t2 = cat(1, Y{10:12});

Train_data = cat(1, X_all{:});
Train_label = cat(1, Y_all{:});

% ----- train SCA using domain 3 as validation set
All_acc = zeros(5, 10, 7);
for i = 1:5
    for j = 1:10
        for k = 1:7
            beta = 0.1+(i-1)*0.2;
            delta = power(10, j-1);
            sigma = power(10, k-4);
            [B, A] = SCA(X_all, Y_all, beta, delta, 0.00001, sigma);

            B = real(B);
            A = real(A);
            [ACC, pre_labels, Zs, Zt] = SCA_test(B, A, X_all, Y_all, X_t1, Y_t1, sigma, 2);

            All_acc(i,j,k) = ACC;
            fprintf('i: %d,j: %d,k: %d, acc: %d \n', i, j, k, ACC);
        end
    end
end

% ----- test using the best parameters
ACC_tr_best = max(All_acc(:));
ind = find(All_acc==max(All_acc(:)));
[m, n, x] = size(All_acc);
[best_i, best_j, best_k] = ind2sub([m n x],ind(1));

best_beta = 0.1+(best_i-1)*0.2;
best_delta = power(10, best_j-1);
best_sigma = power(10, best_k-4);

[B, A] = SCA(X_all, Y_all, best_beta, best_delta, 0.00001, best_sigma);

B = real(B);
A = real(A);
[ACC_test, pre_labels, Z_s, Z_t] = SCA_test(B, A, X_all, Y_all, X_t2, Y_t2, best_sigma, 2);
fprintf('The test accuracy on target domain: %f\n', ACC_test);