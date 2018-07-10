function res_struct=select_randsubs_sametrain(ipmats, behav, numtrain, numiters, thresh, ipmats_ex, behav_ex)

    res_struct=struct();

    for iter = 1:numiters
        
        
        
        looind=numtrain+1;
        k2ind=numtrain/0.5;
        k5ind=numtrain/0.8;
        k10ind=numtrain/0.9;
        
        
        
        fprintf('\n Performing iter # %6.0f of %6.0f \n',iter,numiters);
        totalsubs=size(ipmats,2);
        randinds=randperm(totalsubs);
        randinds=randinds(1:k2ind);
        randipmats=ipmats(:,randinds);
        randbehav=behav(randinds);

        % LOOCV
        [res_struct.loo(iter,1),res_struct.loo(iter,2),res_struct.loo(iter,3),res_struct.loo(iter,4)] = cpm_cv(randipmats(:,1:looind), randbehav(1:looind), looind, thresh);
        % Split half
        [res_struct.k2(iter,1),res_struct.k2(iter,2),res_struct.k2(iter,3),res_struct.k2(iter,4)] = cpm_cv(randipmats(:,1:k2ind), randbehav(1:k2ind), 2, thresh);
        % K = 5
        [res_struct.k5(iter,1),res_struct.k5(iter,2),res_struct.k5(iter,3),res_struct.k5(iter,4)] = cpm_cv(randipmats(:,1:k5ind), randbehav(1:k5ind), 5, thresh);
        % K = 10
        [res_struct.k10(iter,1),res_struct.k10(iter,2),res_struct.k10(iter,3),res_struct.k10(iter,4)] = cpm_cv(randipmats(:,1:k10ind), randbehav(1:k10ind), 10, thresh);
        % External Val
        [fit_pos,fit_neg, pos_mask, neg_mask] = train_cpm(randipmats(:,1:numtrain),randbehav(1:numtrain),thresh);
        
        
        nsubs_ex=size(ipmats_ex,2);
        
        ext_sumpos = sum(ipmats_ex.*repmat(pos_mask,1,nsubs_ex))/2;
        ext_sumneg = sum(ipmats_ex.*repmat(neg_mask,1,nsubs_ex))/2;

        behav_pred_pos_ext = fit_pos(1)*ext_sumpos + fit_pos(2);
        behav_pred_neg_ext = fit_neg(1)*ext_sumneg + fit_neg(2);
        
        
        [Rpos_ext,Ppos_ext]=corr(behav_ex,behav_pred_pos_ext');
        [Rneg_ext,Pneg_ext]=corr(behav_ex,behav_pred_neg_ext');

        res_struct.external(iter,:) = [Rpos_ext, Rneg_ext, Ppos_ext, Pneg_ext];
        
    end

end
