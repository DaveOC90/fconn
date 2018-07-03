function res_struct=nsubs_iter(ipmats,behav,numiters,thresh)

    res_struct=struct();

    for iter = 1:numiters
    
        fprintf('\n Performing iter # %6.0f of %6.0f \n',iter,numiters);

        
        nsubs=size(ipmats,2);
        randinds=randperm(nsubs);
        testinds=randinds(1:400);
        testmats=ipmats(:,testinds);
        testbehav=behav(testinds);
        
        remaining_rand=randinds(401:end);
        
        
        for trainsubs = 25:25:400
            fprintf('\n Training on %6.0f Subs \n',trainsubs);

            traininds=remaining_rand(1:trainsubs);
            trainmats=ipmats(:,traininds);
            trainbehav=behav(traininds);
            
            [fit_pos,fit_neg,pos_mask,neg_mask] = train_cpm(trainmats, trainbehav,thresh);

            test_sumpos = sum(testmats.*repmat(pos_mask,1,400))/2;
            test_sumneg = sum(testmats.*repmat(neg_mask,1,400))/2;

            behav_pred_pos = fit_pos(1)*test_sumpos + fit_pos(2);
            behav_pred_neg = fit_neg(1)*test_sumneg + fit_neg(2);

            [Rpos,Ppos]=corr(testbehav,behav_pred_pos');
            [Rneg,Pneg]=corr(testbehav,behav_pred_neg');
            
            res_struct.(['train' num2str(trainsubs)])(iter,:)=[Rpos Rneg Ppos Pneg];
            
        end    
        
    end


end