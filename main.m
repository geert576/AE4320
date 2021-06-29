%% Main script for playing around
clear all
clf
% [e,its,yRBF,F16_RBF,Z_k] = levenberg(0.001,50);
% plot(e);
method = 'backprop'; % backprop or levenberg
[e,its,yRBF,F16_RBF,Z_k] = feedforward(1e-5,20,method);
%%
plotting(F16_RBF,Z_k);
figure(4)
semilogy(e(5:end))