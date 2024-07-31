%% Arousal pattern test validation dataset
clear,clc
load('validation_36_activation.mat')
load('arousal_brain_activation.mat');
load('validation36_idx.mat')
%overall
svrobj = svr({'C=1', 'optimizer="andre"', kernel('linear')});
dataobj = data('spider data', double(discovery.dat)', discovery.Y);
clear discovery
[~, svrobj] = train(svrobj, dataobj, loss);
weights = get_w(svrobj)';
bias = svrobj.b0;
predicted_ratings = double(data_test.dat)'*weights+bias;

%% overall
pred_Arousal_r_valdiation = corr(data_test.Y,predicted_ratings);

%% within
nsub = 36;
nlevel = 9;
true_ratings = data_test.Y;
subject = repmat(1:nsub, nlevel,1);
subject = subject(:);
subject = subject(validation_label_idx);
within_subj_corrs = zeros(nsub, 1);
within_subj_rmse = zeros(nsub, 1);
    for i = 1:nsub
    subY = true_ratings(subject==i);
    subyfit = predicted_ratings(subject==i);
    within_subj_corrs(i) = corr(subY, subyfit);
    err = subY - subyfit;
    mse = (err' * err)/length(err);
    within_subj_rmse(i) = sqrt(mse);
    end

%% classification
load('predicted_ratings_all_validation36.mat') %included all responses
nrepeat = 1;
nsub = 36;
nlevel = 9;
Accuracy_low_medium_high = zeros(nrepeat, 3);
Accuracy_se_low_medium_high = zeros(nrepeat, 3);
Accuracy_p_low_medium_high = zeros(nrepeat, 3);
Accuracy_d_low_medium_high = zeros(nrepeat, 3);
for n = 1:nrepeat
    PE = predicted_ratings_all(:,n); % 
    PE = reshape(PE, [9, nsub])';
    PE_low = nanmean(PE(:, 1:3),2);
    PE_medium = nanmean(PE(:,4:6), 2);
    PE_high = nanmean(PE(:, 7:9), 2);
    no_rating1 =  find(isnan(PE_medium));
    rowsToKeep = true(size(PE_medium, 1), 1);
    rowsToKeep(no_rating1) = false;
    PE_low = PE_low(rowsToKeep, :);
    PE_medium = PE_medium(rowsToKeep, :);
    PE_high = PE_high(rowsToKeep, :);
    
    no_rating2 =  find(isnan(PE_high));
    rowsToKeep = true(size(PE_medium, 1), 1);
    rowsToKeep(no_rating2) = false;
    PE_low = PE_low(rowsToKeep, :);
    PE_medium = PE_medium(rowsToKeep, :);
    PE_high = PE_high(rowsToKeep, :);
    n_sub = length(PE_low);
    % low vs. meduim
    ROC = roc_plot([PE_medium;PE_low], [ones(n_sub,1);zeros(n_sub,1)], 'twochoice');
    Accuracy_low_medium_high(n,1) = ROC.accuracy;
    Accuracy_se_low_medium_high(n,1) = ROC.accuracy_se;
    Accuracy_p_low_medium_high(n,1) = ROC.accuracy_p;
    Accuracy_d_low_medium_high(n,1) = ROC.Gaussian_model.d_a;
    % medium vs. high
    ROC = roc_plot([PE_high;PE_medium], [ones(n_sub,1);zeros(n_sub,1)], 'twochoice');
    Accuracy_low_medium_high(n,2) = ROC.accuracy;
    Accuracy_se_low_medium_high(n,2) = ROC.accuracy_se;
    Accuracy_p_low_medium_high(n,2) = ROC.accuracy_p;
    Accuracy_d_low_medium_high(n,2) = ROC.Gaussian_model.d_a;
    % low vs. high
    ROC = roc_plot([PE_high;PE_low], [ones(n_sub,1);zeros(n_sub,1)], 'twochoice');
    Accuracy_low_medium_high(n,3) = ROC.accuracy;
    Accuracy_se_low_medium_high(n,3) = ROC.accuracy_se;
    Accuracy_p_low_medium_high(n,3) = ROC.accuracy_p;
    Accuracy_d_low_medium_high(n,3) = ROC.Gaussian_model.d_a;
end