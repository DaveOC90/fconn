fpaths=dir('indv_phase/*.mat');
load('wm_meanphasecon_pmats.mat')


nsubs=400;

subpmats=subpmats(1:nsubs);
wm_mask=~isnan(subpmats);

subpmats_mask=subpmats(wm_mask);
numfs=length(fpaths);
%fpaths=fpaths(wm_mask(1:numfs),:);

numiters=100;


for tp = 1:numfs
    fprintf(['timepoint ' num2str(tp) '\n'])
%    ipmats=zeros(268,268,numfs);
    fprintf('loading data \n')
%     reverseStr='';
%     for j = 1:numfs
%         load(['indv_phase/' fpaths(j).name]);
%         ipmats(:,:,j)=mean(wm_pc_indv(i:i+4,:,:),1);
%         percentDone = 100 * j / numfs;
%         msg = sprintf('%1d file(s) read, percent done: %3.1f \n', j, percentDone);
%         fprintf([reverseStr, msg]);
%         reverseStr = repmat(sprintf('\b'), 1, length(msg)); 
%     end
    load(['indv_phase/' fpaths(tp).name])
    ipmats=reshape(wm_pc_indv,nsubs,268^2)';
    ipmats_mask=ipmats(:,wm_mask);
    
    for iter = 1:numiters
        fprintf(['running cpm, iteration: ' num2str(iter) '\n'])
        [Rpos,Rneg,Ppos,Pneg,Rmsepos,Rmseneg,test_behav_gather,behav_pred_pos,behav_pred_neg,pos_mask_gather,neg_mask_gather] = cpm_cv(ipmats_mask, subpmats_mask', 2, 0.01, 1,false);
      
        Rpos_gather(tp,iter)=Rpos;
    end
end

% 
% rpred_fs={'dCPM_1tp_400subs.mat','dCPM_5tp_400subs.mat','dCPM_10tp_400subs.mat','dCPM_20tp_400subs.mat','dCPM_30tp_400subs.mat'};
% 
% for kk = 1:numel(rpred_fs)
%     load(rpred_fs{kk})
% end
% 
% 
% r_preds=zeros(5,405);
% r_preds(:,:)=nan;
% 
% r_preds(1,:)=Rpos_gather_1tp;
% r_preds(2,6:405)=Rpos_gather_5tp;
% r_preds(3,11:405)=Rpos_gather_10tp;
% r_preds(4,20:405)=Rpos_gather_20tp;
% r_preds(5,30:405)=Rpos_gather_30tp;
% 
% % plot(r_preds(1,:)');hold on;plot(r_preds(5,:)')
% 
% 
% 
% 
%eventfile='../LEiDA/corrmats_static_le/HCP-WM-LR-EPrime/100307/100307_3T_WM_run2_TAB_filtered.csv';
eventfile='HCP-WM-LR-EPrime/100307/100307_3T_WM_run2_TAB_filtered.csv';
events=readtable(eventfile);
event_str=strings(405,1);
event_str(events.VolumeAssignment)=events.EventCol;
event_str(event_str == string('')) = NaN;
event_str=fillmissing(event_str,'previous');


event_str=strrep(event_str,'-Body','');
event_str=strrep(event_str,'-Face','');
event_str=strrep(event_str,'-Tools','');
event_str=strrep(event_str,'-Place','');

[a b c] = unique(event_str);
% 
% titles={'Instantaneous Phase Connectivity',
%     'Mean of 5 Points',
%     'Mean of 10 Points',
%     'Mean of 20 Points',
%     'Mean of 30 Points'};
% 
% for ind = 1:5
%     fig=figure;
%     set(fig, 'Position', [50 50 1100 550])
%     
%     imagesc(1:405,-1:1,c');
%     colorbar('YTickLabel',a)
%     %colormap('hsv')
%     hold on;
%     plot(r_preds(ind,:)','k','LineWidth',1);
%     set(gca,'YDir','normal');
%     ylim([-0.2 0.3]);
%     title(titles{ind})
%     saveas(fig,['/home/dmo39/dCPM_' titles{ind} '.png'])
% end
% 
