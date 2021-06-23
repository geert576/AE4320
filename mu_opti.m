clear all
close all
clc


testlength = 13;
hidden_layer_size = 20;
testruns = 10;
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

% Determine best mu to start with


%% 
%semilogy(elist);
loglog(mulist,elist);



