%% Feed-forward neural network, using 2 different optimization techiques
function [errorlist,its,yFF,F16_FF,Z_k] = feedforward(mu,hidden_layer_size,method)
load('F16traindata_CMabV_2021.mat', 'Cm', 'Z_k', 'U_k');

% Estimating the F16 training data with a RBF NN
F16_FF = [];
[Xtrain,Xtest,Ytrain,Ytest] = preprocess(Z_k,Cm,0.9,true);
alpha = Xtrain(:,1);
beta  = Xtrain(:,2);
V     = Xtrain(:,3);
Z_kn  = Ytrain;
Z_k   = cat(1,Xtrain,Xtest);
% Init
weight_output = randn(1);
alpha_w   = randn(1)*ones(hidden_layer_size,1);
beta_w    = randn(1)*ones(hidden_layer_size,1);
V_w       = randn(1)*ones(hidden_layer_size,1);
bias_inw  = randn(1)*ones(hidden_layer_size,1);
bias_outw = randn(1)*ones(1,hidden_layer_size);
bias_in   = randn(hidden_layer_size,1);
bias_out  = randn(1,1);
w_init_in = cat(2,alpha_w,beta_w,V_w);
w_init_output = ones(1,hidden_layer_size)*weight_output;
F16_FF.IW = w_init_in;
F16_FF.LW = w_init_output;
F16_FF.range     = ones(3,2);
F16_FF.b{1,1}    = bias_in;
F16_FF.b{2,1}    = bias_out;
F16_FF.epochs    = 10000;
F16_RBF.goal     = 0;
F16_RBF.min_grad = 1e-12;

F16_FF.name{1,1} = 'feedforward';
F16_FF.trainFunct{1,1} = 'tansig';
F16_FF.trainFunct{2,1} = 'purelin';

yFF = simNet(F16_FF,Xtrain');
eold      = Ytrain' -  yFF.Y2;
wold      = cat(2,w_init_in,w_init_output');
e         = mse(eold);
errorlist = 100*ones(F16_FF.epochs,1);
mse_init  = e; 
for epoch =1:F16_FF.epochs
    
    if method == 'backprop'
        [dw_in,dw_out] = backpropagation(eold,Xtrain,yFF,F16_FF);

        wold = wold - mu*cat(2,dw_in,dw_out');
        F16_FF.IW = wold(:,1:3);
        F16_FF.LW = wold(:,4)';
        yFF = simNet(F16_FF,Xtrain');
        eold      = Ytrain' -  yFF.Y2;
    elseif method == 'levenberg'
        
    end
    % disp(mse(eold));
    errorlist(epoch) = mse(eold);
    e_grad = gradient(errorlist(1:epoch));
    
    % Check the stopping conditions
    if mse(eold) == F16_RBF.goal
        disp('Goal reached')
        break
    elseif (epoch > 2) && (abs(e_grad(end)) <= F16_RBF.min_grad)
        disp('Minimum gradient reached')
        break
    elseif epoch == F16_FF.epochs
        disp('Max epoch reached');
    elseif isnan(mse(eold))
        disp('NaN value')
        break
    end
end
its = epoch;
disp(mse(eold));
end