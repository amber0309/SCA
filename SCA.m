function [test_accuracy, predicted_labels, Zs, Zt] = SCA(X_s_cell, Y_s_cell, X_t, Y_t, params)
%{
Implementation of Scatter Component Analysis (SCA) proposed in [1]

INPUT(params is optional):
  X_s_cell          - cell of (n_s*d) matrix, each matrix corresponds to the instance features of a source domain
  Y_s_cell          - cell of (n_s*1) matrix, each matrix corresponds to the instance labels of a source domain
  X_t               - (n_t*d) matrix, rows correspond to instances and columns correspond to features
  Y_t               - (n_t*1) matrix, each row is the class label of corresponding instances in X_t
  [params]          - params.beta:      vector of validated values of beta
                      params.delta:     vector of validated values of delta
                      params.k_list:    vector of validated dimension of the transformed space
                      params.X_v:       (n_v*d) matrix of instance features of validation set (use the source instances if not provided)
                      params.Y_v:       (n_v*1) matrix of instance labels of validation set (use the source instances if not provided)
                      params.verbose:   if true, show the validation accuracy of each parameter setting

OUTPUT:
  test_accuracy     - test accuracy on target instances
  predicted_labels  - predicted labels of target instances
  Zs                - projected source domain instances
  Zt                - projected target domain instances

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-02

Reference
[1] Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). 
    Scatter component analysis: A unified framework for domain 
    adaptation and domain generalization. IEEE transactions on pattern 
    analysis and machine intelligence, 39(7), 1414-1430.
%}

    if nargin < 4
        error('Error. \nOnly %d input arguments! At least 4 required', nargin);
    elseif nargin == 4
        % default params values
        beta = [0.1 0.3 0.5 0.7 0.9];
        delta = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6];
        k_list = [2];
        X_v = cat(1, X_s_cell{:});
        Y_v = cat(1, Y_s_cell{:});
        verbose = false;
    elseif nargin == 5
        if ~isfield(params, 'beta')
            beta = [0.1 0.3 0.5 0.7 0.9];
        else
            beta = params.beta;
        end
        
        if ~isfield(params, 'delta')
            delta = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6];
        else
            delta = params.delta;
        end

        if ~isfield(params, 'k_list')
            k_list = [2];
        else
            k_list = params.k_list;
        end

        if ~isfield(params, 'verbose')
            verbose = false;
        else
            verbose = params.verbose;
        end

        if ~isfield(params, 'X_v')
            X_v = cat(1, X_s_cell{:});
            Y_v = cat(1, Y_s_cell{:});
        else
            if ~isfield(params, 'Y_v')
                error('Error. Labels of validation set needed!');
            end
            X_v = params.X_v;
            Y_v = params.Y_v;
        end
    end

    % ----- training phase
    % ----- ----- source domains
    X_s = cat(1, X_s_cell{:});
    Y_s = cat(1, Y_s_cell{:});
    fprintf('Number of source domains: %d, Number of classes: %d.\n', length(X_s_cell), length(unique(Y_s)) );
    fprintf('Validating hyper-parameters ...\n');

    dist_s_s = pdist2(X_s, X_s);
    dist_s_s = dist_s_s.^2;
    sgm_s = compute_width(dist_s_s);
    % ----- ----- validation set
    dist_s_v = pdist2(X_s, X_v);
    dist_s_v = dist_s_v.^2;
    sgm_v = compute_width(dist_s_s);

    n_s = size(X_s, 1);
    n_v = size(X_v, 1);
    H_s = eye(n_s) - ones(n_s)./n_s;
    H_v = eye(n_v) - ones(n_v)./n_v;
        
    K_s_s = exp(-dist_s_s./(2 * sgm_s * sgm_s));
    K_s_v = exp(-dist_s_v./(2 * sgm_v * sgm_v));
    K_s_v_bar = H_s * K_s_v * H_v;
    [P, T, D, Q, K_s_s_bar] = SCA_terms(K_s_s, X_s_cell, Y_s_cell);

    acc_mat = zeros(length(k_list), length(beta), length(delta));
    for i = 1:length(beta)
        cur_beta = beta(i);
        for j = 1:length(delta)
            cur_delta = delta(j);
            [B, A] = SCA_trans(P, T, D, Q, K_s_s_bar, cur_beta, cur_delta, 1e-5);

            for k = 1:length(k_list)
                [acc, ~, ~, ~] = SCA_test(B, A, K_s_s_bar, K_s_v_bar, Y_s, Y_v, k_list( k ) );
                acc_mat(k, i, j) = acc;
                if verbose
                    fprintf('beta: %f, delta: %f, acc: %f\n', cur_beta, cur_delta, acc);
                end
            end
        end
    end

    fprintf('Validation done! Classifying the target domain instances ...\n');
    % ----- test phase
    % ----- ----- get validated parameters
    acc_tr_best = max( acc_mat(:) );
    ind = find( acc_mat == acc_tr_best );
    [k, i, j] = size( acc_mat );
    [best_k, best_i, best_j] = ind2sub([k, i, j], ind(1));

    best_beta = beta(best_i);
    best_delta = delta(best_j);
    best_k = k_list(best_k);
    
    % ----- ----- test on the target domain
    dist_s_t = pdist2(X_s, X_t);
    dist_s_t = dist_s_t.^2;
    sgm = compute_width(dist_s_t);
    K_s_t = exp(-dist_s_t./(2 * sgm * sgm));
    n_s = size(X_s, 1);
    H_s = eye(n_s) - ones(n_s)./n_s;
    n_t = size(X_t, 1);
    H_t = eye(n_t) - ones(n_t)./n_t;
    K_s_t_bar = H_s * K_s_t * H_t;

    [B, A] = SCA_trans(P, T, D, Q, K_s_s_bar, best_beta, best_delta, 1e-5);
    [test_accuracy, predicted_labels, Zs, Zt] = SCA_test(B, A, K_s_s_bar, K_s_t_bar, Y_s, Y_t, best_k );
    fprintf('Test accuracy: %f\n', test_accuracy);

end