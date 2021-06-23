clear all
close all
clc


testlength = 13;
hidden_layer_size = 20;
testruns = 5;
mulist = zeros(testlength,1);
mu     = 1e-8;
for i = 1:length(mulist)
    mulist(i) = mu;
    mu = mu*10;
end

elist   = zeros(testlength,1);
itslist = zeros(testlength,1);
for i = 1:length(mulist)
    mu = mulist(i);
    etotal   = 0;
    itstotal = 0;
    for j = 1:testruns
        [e,its,yRBF] = levenberg(mu,hidden_layer_size);
        e_mse = mse(e);
        etotal   = etotal + e_mse;
        itstotal = itstotal + its;
    end
    e_mse = etotal/testruns;
    its   = itstotal/testruns;
    elist(i)   = e_mse;
    itslist(i) = its;
end
% 
semilogy(elist);
loglog(mulist,elist);

%%
% Determine best mu to start with
[min,idx] = min(elist);
mu_min = mulist(idx);
elist_neurons   = zeros(testlength,1);
itslist_neurons = zeros(testlength,1);
neurons = [1:1:150];
for i = 1:length(neurons)
    hidden_layer_size = neurons(i);
    etotal   = 0;
    itstotal = 0;
    for j = 1:testruns
        [e,its,yRBF] = levenberg(mu_min,hidden_layer_size);
        while mse(e) > 0.03
            % Only use the runs that converged properly
            [e,its,yRBF] = levenberg(mu_min,hidden_layer_size);
        end
        etotal = etotal + mse(e);
        itstotal = itstotal + its;
    end
    elist_neurons(i)   = etotal/testruns;
    itslist_neurons(i) = itstotal/testruns;
    disp(i);
end




