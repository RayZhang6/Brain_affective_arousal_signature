clear,clc
load('arousal_brain_activation.mat')
load('discovery_arousal_label.mat')

nsub = 60;
nrepeat = 10; % for 10X10 cross-validation
nlevel = 9; % 9 ratings
predicted_ratings = zeros(length(discovery.Y),nrepeat);
for i = 1:10
    idx = GenerateCV(nsub, nlevel, nrepeat); 
    CVindex(:,i) = idx(discovery_label_idx);
end
save('CV_for_replication.mat','CVindex');
load('CV_for_replication.mat')

for repeat = 1:nrepeat
    [~, stats] = predict(discovery,  'algorithm_name', 'cv_svr', 'nfolds', CVindex(:,repeat), 'error_type', 'mse');
    predicted_ratings(:, repeat) = stats.yfit;
end

%% overall (between- and within-subjects) prediction-outcome correlations
true_ratings = discovery.Y;
prediction_outcome_corrs = corr(true_ratings, predicted_ratings);

%% Within-subject prediction-outcome correlations
subject = repmat(1:nsub, nlevel,1);
subject = subject(:);
subject = subject(discovery_label_idx);

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


