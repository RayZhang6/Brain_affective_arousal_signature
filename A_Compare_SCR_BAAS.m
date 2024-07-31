%% Compare SCR and BAAS

%% SCR predicts Study 19

clear,clc

Arousal = fmri_data('SCR_decoder.nii', 'SCR_GM_mask.nii');
load('Study19_CS_data.mat')
Arousalpe = double(Zhou_CSup_CSm_data.dat'*Arousal.dat);
roc_SCR = roc_plot(Arousalpe, [ones(58,1);zeros(58,1)],'color',[236 93 59]/255,'twochoice');

set(gca,'FontName','Helvetica Neue','FontSize',12, 'LineWidth', 1)
set(gca,'FontSize',20);
set(gca,'linewidth', 2)
set(gca, 'XTick', 0:1)
set(gcf, 'Color', 'w');
set(gca,'YLim', [0 1.05], 'ytick',0:0.2:1, 'XLim', [0 1.05], 'xtick',0:0.2:1, 'tickdir', 'out');
box off
set(gcf, 'PaperPosition', [0 0 5 6]);
export_fig A_SCR_predict_study19 -tiff -r400


%% SCR predicts Study 20 SCL

clear,clc

load('Study9_SCL_Lucy_n43.mat')
load('SCR_brain_activation.mat');
discovery = scr_data;
data_test = resample_space(SCL,discovery);
svrobj = svr({'C=1', 'optimizer="andre"', kernel('linear')});
dataobj = data('spider data', double(discovery.dat)', discovery.Y);
clear discovery
[~, svrobj] = train(svrobj, dataobj, loss);
weights = get_w(svrobj)';
bias = svrobj.b0;
predicted_ratings = double(data_test.dat)'*weights+bias;

high_idx = find (data_test.Y == 5|data_test.Y == 4);
low_idx = find (data_test.Y == 1|data_test.Y == 2);
high1(:,1) = predicted_ratings(high_idx(1:43));
high1(:,2) = predicted_ratings(high_idx(44:86));
low1(:,1) = predicted_ratings(low_idx(1:43));
low1(:,2) = predicted_ratings(low_idx(44:86));
high = mean(high1,2);
low = mean(low1,2);
ROC = roc_plot([high;low], [ones(43,1);zeros(43,1)],'color',[28 16 70]/255,'twochoice');
set(gca,'FontName','Helvetica Neue','FontSize',12, 'LineWidth', 1)
set(gca,'FontSize',20);
set(gca,'linewidth', 2)
set(gca, 'XTick', 0:1)
set(gcf, 'Color', 'w');
set(gca,'YLim', [0 1.05], 'ytick',0:0.2:1, 'XLim', [0 1.05], 'xtick',0:0.2:1, 'tickdir', 'out');
box off
set(gcf, 'PaperPosition', [0 0 5 6]);


%% BAS predicts Study 20 SCL

clear,clc

load('Study9_SCL_Lucy_n43.mat') % downloaded in Liu et al., 2023 NC
load('arousal_brain_activation.mat');
l_scr_data = resample_space(SCL,discovery);
data_test = l_scr_data;
svrobj = svr({'C=1', 'optimizer="andre"', kernel('linear')});
dataobj = data('spider data', double(discovery.dat)', discovery.Y);
clear discovery
[~, svrobj] = train(svrobj, dataobj, loss);
weights = get_w(svrobj)';
bias = svrobj.b0;
predicted_ratings = double(data_test.dat)'*weights+bias;

high_idx = find (data_test.Y == 5|data_test.Y == 4);
low_idx = find (data_test.Y == 1|data_test.Y == 2);
high1(:,1) = predicted_ratings(high_idx(1:43));
high1(:,2) = predicted_ratings(high_idx(44:86));
low1(:,1) = predicted_ratings(low_idx(1:43));
low1(:,2) = predicted_ratings(low_idx(44:86));
high = mean(high1,2);
low = mean(low1,2);
ROC = roc_plot([high;low], [ones(43,1);zeros(43,1)],'color',[28 16 70]/255,'twochoice');
set(gca,'FontName','Helvetica Neue','FontSize',12, 'LineWidth', 1)
set(gca,'FontSize',20);
set(gca,'linewidth', 2)
set(gca, 'XTick', 0:1)
set(gcf, 'Color', 'w');
set(gca,'YLim', [0 1.05], 'ytick',0:0.2:1, 'XLim', [0 1.05], 'xtick',0:0.2:1, 'tickdir', 'out');
box off
set(gcf, 'PaperPosition', [0 0 5 6]);

%% SCR predicts Study 20 anxious feeling

clear,clc

load('Study9_Anxiety_decoder_noshock_n43.mat') % of note: we keep the same number with SCL data
load('SCR_data_mask.mat');
discovery = scr_data;
data_test = resample_space(indep_new,discovery);
svrobj = svr({'C=1', 'optimizer="andre"', kernel('linear')});
dataobj = data('spider data', double(discovery.dat)', discovery.Y);
clear discovery
[~, svrobj] = train(svrobj, dataobj, loss);
weights = get_w(svrobj)';
bias = svrobj.b0;
predicted_ratings = double(data_test.dat)'*weights+bias;

pred =nan(215,1);
pred(sub_5043_lable) = predicted_ratings;
pred = reshape(pred,5,43)';
low = nanmean(pred(:,1:2),2);
high = nanmean(pred(:,4:5),2);
ROC = roc_plot([high;low], [ones(43,1);zeros(43,1)],'color',[182 54 121]/255,'twochoice');

%% BAS predict Study 20 anxious feeling
clear,clc
load('Study3_Anxiety_decoder_noshock_n43.mat')
load('arousal_brain_activation.mat');
% discovery = discovery;
data_test = resample_space(indep_new, discovery);
svrobj = svr({'C=1', 'optimizer="andre"', kernel('linear')});
dataobj = data('spider data', double(discovery.dat)', discovery.Y);
clear discovery
[~, svrobj] = train(svrobj, dataobj, loss);
weights = get_w(svrobj)';
bias = svrobj.b0;
predicted_ratings = double(data_test.dat)'*weights+bias;

pred =nan(215,1);
pred(sub_5043_lable) = predicted_ratings;
pred = reshape(pred,5,43)';
low = nanmean(pred(:,1:2),2);
high = nanmean(pred(:,4:5),2);
ROC = roc_plot([high;low], [ones(43,1);zeros(43,1)],'color',[182 54 121]/255,'twochoice');
