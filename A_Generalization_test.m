clear,clc

%% generalization in Study3 (negative videos)

gray_matter_mask = 'GM_mask.nii'; 
Arousal = fmri_data('BAAS.nii', gray_matter_mask);

%% disgust vs neutral
dis_neu = spm_select('ExtFPList', ['E:\Arousal_MRI_data\Negative_veideo_dataset\Contrast_discovery_video_D_N\'], ['^r', '.*nii$']);
dis_neu = fmri_data (dis_neu,gray_matter_mask);
Arousalpe = double(dis_neu.dat'*Arousal.dat); % HP pattern expression for cona vs conb
roc_D_N = roc_plot(Arousalpe, [zeros(78,1);ones(78,1)],'color',[128 39 3]/255,'twochoice'); 

%% fear vs neutral
fear_neu = spm_select('ExtFPList', ['E:\Arousal_MRI_data\Negative_veideo_dataset\Contrast_discovery_video_F_N\'], ['^r', '.*nii$']);
fear_neu = fmri_data (fear_neu,gray_matter_mask);
Arousalpe = double(fear_neu.dat'*Arousal.dat); % HP pattern expression for cona vs conb
roc_F_N = roc_plot(Arousalpe, [zeros(78,1);ones(78,1)],'color',[182 60 2]/255,'twochoice');

%% fear vs disgust
dis_fear = spm_select('ExtFPList', ['E:\Arousal_MRI_data\Negative_veideo_dataset\Contrast_discovery_video_D_F\'], ['^r', '.*nii$']);
dis_fear = fmri_data (dis_fear,gray_matter_mask);
Arousalpe = double(dis_fear.dat'*Arousal.dat); % HP pattern expression for cona vs conb
roc_D_F = roc_plot(Arousalpe, [zeros(78,1);ones(78,1)],'color',[182 60 2]/255,'twochoice'); 

set(gca,'FontName','Helvetica Neue','FontSize',12, 'LineWidth', 1)
set(gca,'FontSize',20);
set(gca,'linewidth', 2)
set(gca, 'XTick', 0:1)
set(gcf, 'Color', 'w');
set(gca,'YLim', [0 1.05], 'ytick',0:0.2:1, 'XLim', [0 1.05], 'xtick',0:0.2:1, 'tickdir', 'out');
box off
set(gcf, 'PaperPosition', [0 0 5 6]);
% export_fig A_BAAS_predict_Negative_videos -tiff -r400

%% We used the same code to test generalization datasets from studies 4 - 6 and 17.

%%  generalization in Studies 7 - 16

load('alldata.mat'); % this file could be downloaded in Kragel et al., 2023 NHB
arousal = fmri_data('BAAS_SPM_nearestNeighbor.nii', 'mask_Phil.nii');
arousal = resample_space(arousal, alldat);
PE = alldat.dat'*arousal.dat;
study_data = {};
clear p
for i = 1:10
[~, p(i), tstat{i}] = ttest(PE(study==i));
study_data{i,1} = PE(study==i);
end
p

