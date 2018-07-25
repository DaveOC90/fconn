function [Rpos,Rneg,Ppos,Pneg,Rmsepos,Rmseneg] = cpm_cv(ipmats, behav, kfolds, thresh, popvar)
    
    nsubs=size(ipmats,2);
    randinds=randperm(nsubs);
    ksample=floor(nsubs/kfolds);
   
    
    behav_pred_pos=zeros(kfolds,ksample);
    behav_pred_neg=zeros(kfolds,ksample);
    test_behav_gather=zeros(kfolds,ksample);
    
    for leftout = 1:kfolds
        fprintf('\n Performing fold # %6.0f of %6.0f \n',leftout,kfolds);
        tic
        if kfolds == nsubs
            testinds=randinds(leftout);
            traininds=setdiff(randinds,testinds);            
        else
            si=1+((leftout-1)*ksample);        
            fi=si+ksample-1;

            testinds=randinds(si:fi);
            traininds=setdiff(randinds,testinds);
        
        end
        
        % leave out subject from matrices and behavior

        train_vcts = ipmats(:,traininds);
        train_behav = behav(traininds);
        
        [fit_pos,fit_neg,pos_mask,neg_mask] = train_cpm(train_vcts, train_behav,thresh);

        test_vcts = ipmats(:,testinds);
        test_behav = behav(testinds);
        test_behav_gather(leftout,:)=test_behav;
       
        nsubs_inner=size(test_vcts,2);
 
        test_sumpos = sum(test_vcts.*repmat(pos_mask,1,nsubs_inner))/2;
        test_sumneg = sum(test_vcts.*repmat(neg_mask,1,nsubs_inner))/2;

        behav_pred_pos(leftout,:) = fit_pos(1)*test_sumpos + fit_pos(2);
        behav_pred_neg(leftout,:) = fit_neg(1)*test_sumneg + fit_neg(2);
        
        toc
    end

    behav_pred_pos=reshape(behav_pred_pos,[],1);
    behav_pred_neg=reshape(behav_pred_neg,[],1);
    test_behav_gather=reshape(test_behav_gather,[],1);
    
    [Rpos,Ppos]=corr(test_behav_gather,behav_pred_pos);
    [Rneg,Pneg]=corr(test_behav_gather,behav_pred_neg);
    
    mse_pos=mean((test_behav_gather-behav_pred_pos).^2);
    mse_neg=mean((test_behav_gather-behav_pred_neg).^2);
    
    Rmsepos=sqrt(1-mse_pos/popvar);
    Rmseneg=sqrt(1-mse_neg/popvar);
    
end
