clear,clc
nsub = 25;
nrepeat = 5; % for 5X5 cross-validation
nlevel = 6; % 6 ratings
load('SCR_brain_activation.mat')
load('SCR_discovery_lable_idx.mat')

discovery = scr_data;
predicted_ratings = zeros(length(discovery.Y),nrepeat);
for i = 1:nrepeat
    CVindex = GenerateCV(nsub, nlevel, i); 
    CVindex = CVindex(discovery_single_label);
    [~, stats] = predict(discovery,  'algorithm_name', 'cv_svr', 'nfolds', CVindex, 'error_type', 'mse');
    predicted_ratings(:, i) = stats.yfit;
end
%% overall (between- and within-subjects) prediction-outcome correlations
true_ratings = discovery.Y;
prediction_outcome_corrs = corr(true_ratings, predicted_ratings);

%% Within-subject prediction-outcome correlations
subject = repmat(1:nsub, nlevel,1);
subject = subject(:);

within_subj_corrs = zeros(nsub, nrepeat);
within_subj_rmse = zeros(nsub, nrepeat);
for n = 1:nrepeat
    for i = 1:nsub
    subY = true_ratings(subject==i);
    subyfit = predicted_ratings(subject==i, n);
    within_subj_corrs(i, n) = corr(subY, subyfit);
    err = subY - subyfit;
    mse = (err' * err)/length(err);
    within_subj_rmse(i, n) = sqrt(mse);
    end
end
