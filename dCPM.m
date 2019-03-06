fpaths=(ls('D:/*.mat'));
load('../LEiDA/corrmats_static_le/code/wm_meanphasecon_pmats.mat')
wm_mask=~isnan(subpmats);



subpmats_mask=subpmats(wm_mask);

fpaths=fpaths(wm_mask(1:numfs),:);
numfs=length(fpaths);

for i = 32:401
    fprintf(['timepoint ' num2str(i) '\n'])
    ipmats=zeros(268,268,numfs);
    fprintf('loading data \n')
    reverseStr='';
    for j = 1:numfs
        load(['D:\' fpaths(j,:)]);
        ipmats(:,:,j)=mean(wm_pc_indv(i:i+4,:,:),1);
        percentDone = 100 * j / numfs;
        msg = sprintf('%1d file(s) read, percent done: %3.1f \n', j, percentDone);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg)); 
    end
    ipmats=reshape(ipmats,268^2,numfs);
    fprintf('running cpm\n')
    [Rpos,Rneg,Ppos,Pneg,Rmsepos,Rmseneg,test_behav_gather,behav_pred_pos,behav_pred_neg,pos_mask_gather,neg_mask_gather] = cpm_cv(ipmats, subpmats_mask(1:numfs)', 2, 0.01, 1,false);

    thing(i)=Rpos;
end

eventfile='../LEiDA/corrmats_static_le/HCP-WM-LR-EPrime/100307/100307_3T_WM_run2_TAB_filtered.csv';
events=readtable(eventfile);
event_str=strings(405,1);
event_str(events.VolumeAssignment)=events.EventCol;
event_str(event_str == "") = missing;
event_str=fillmissing(event_str,'previous');


event_str=strrep(event_str,'-Body','');
event_str=strrep(event_str,'-Face','');
event_str=strrep(event_str,'-Tools','');
event_str=strrep(event_str,'-Place','');

[a b c] = unique(event_str);


imagesc(1:405,-1:1,c');
colorbar('YTickLabel',a)
%colormap('hsv')
hold on;
plot(instaCPM_RPos,'k','LineWidth',1);
set(gca,'YDir','normal');
ylim([-0.2 0.4]);

