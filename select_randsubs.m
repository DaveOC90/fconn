function res_struct=select_randsubs(ipmats, behav, numsubs, numiters, thresh)

    res_struct=struct();

    for iter = 1:numiters
        
        fprintf('\n Performing iter # %6.0f of %6.0f \n',iter,numiters);
        randinds=randperm(numsubs);
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
        
        
    end

end
