function res_struct=select_randsubs(ipmats, behav, numsubs, numiters, thresh, ipmats_ex, behav_ex)

    res_struct=struct();

    for iter = 1:numiters
        
        fprintf('\n Performing iter # %6.0f of %6.0f \n',iter,numiters);
        totalsubs=size(ipmats,2);
        randinds=randperm(totalsubs);
        randinds=randinds(1:numsubs);
        randipmats=ipmats(:,randinds);
        randbehav=behav(randinds);

        % LOOCV
         [res_struct.loo(iter,1),res_struct.loo(iter,2),res_struct.loo(iter,3),res_struct.loo(iter,4)] = cpm_cv(randipmats, randbehav, numsubs, thresh);
        % Split half
        [res_struct.k2(iter,1),res_struct.k2(iter,2),res_struct.k2(iter,3),res_struct.k2(iter,4)] = cpm_cv(randipmats, randbehav, 2, thresh);
        % K = 5
        [res_struct.k5(iter,1),res_struct.k5(iter,2),res_struct.k5(iter,3),res_struct.k5(iter,4)] = cpm_cv(randipmats, randbehav, 5, thresh);
        % K = 10
        [res_struct.k10(iter,1),res_struct.k10(iter,2),res_struct.k10(iter,3),res_struct.k10(iter,4)] = cpm_cv(randipmats, randbehav, 10, thresh);
        % External Val
        [fit_pos,fit_neg, pos_mask, neg_mask] = train_cpm(randipmats,randbehav,thresh);
        
        
        nsubs_ex=size(ipmats_ex,2);
        
        ext_sumpos = sum(ipmats_ex.*repmat(pos_mask,1,nsubs_ex))/2;
        ext_sumneg = sum(ipmats_ex.*repmat(neg_mask,1,nsubs_ex))/2;

        behav_pred_pos_ext = fit_pos(1)*ext_sumpos + fit_pos(2);
        behav_pred_neg_ext = fit_neg(1)*ext_sumneg + fit_neg(2);
        
        test_sumpos = sum(test_vcts.*)/2;
        test_sumneg = sum(test_vcts.*)/2;
        
        
        [Rpos_ext,Ppos_ext]=corr(behav_ex,behav_pred_pos_ext');
        [Rneg_ext,Pneg_ext]=corr(behav_ex,behav_pred_neg_ext');

        res_struct.external(iter,:) = [Rpos_ext, Rneg_ext, Ppos_ext, Pneg_ext];
        
    end

end
