clear all
clc

addpath('./modules');
load('./syn_data/data.mat');

% ----- parameters
% target / all / source domains
tgt_dm = [5];
val_dm = [3 4];
src_dm = [1 2];

data_cell = XY_cell;
X_t = data_cell{tgt_dm(1)}(:, 1:2);
Y_t = data_cell{tgt_dm(1)}(:, 3);

% ----- training data
X_s_cell = cell(1,length(src_dm));
Y_s_cell = cell(1,length(src_dm));    
for idx = 1:length(src_dm)
    cu_dm = src_dm(1, idx);
    X_s_cell{idx} = data_cell{cu_dm}(:, 1:2);
    Y_s_cell{idx} = data_cell{cu_dm}(:, 3);
end
% ----- validation data
X_v = [];
Y_v = [];
for idx = 1:length(val_dm)
    cu_dm = val_dm(1, idx);
    X_v = [X_v; data_cell{cu_dm}(:, 1:2)];
    Y_v = [Y_v; data_cell{cu_dm}(:, 3)];
end

params.X_v = X_v;
params.Y_v = Y_v;
params.verbose = true;
[test_accuracy, predicted_labels, Zs, Zt] = SCA(X_s_cell, Y_s_cell, X_t, Y_t, params);
