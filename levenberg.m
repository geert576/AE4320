function [e,its,yRBF,F16_RBF,Z_k] = levenberg(mu,hidden_layer_size)
load('F16traindata_CMabV_2021.mat', 'Cm', 'Z_k', 'U_k');
F16_RBF = [];

F16_RBF.mu       = mu;
F16_RBF.min_grad = 1e-17;
F16_RBF.epochs   = 1000;
F16_RBF.goal     = 0;

adapt_rate = 10;
mu_max = 5e30;
mu_min = 5e-10;

% Estimating the F16 training data with a RBF NN
[Xtrain,Xtest,Ytrain,Ytest] = preprocess(Z_k,Cm,0.9,true);
Z_kt  = Xtrain;
Cmt   = Ytrain;
Z_k   = cat(1,Xtrain,Xtest);
Cm    = cat(1,Ytrain,Ytest);
% Init
F16_RBF.range = ones(3,2);

F16_RBF.name{1,1} = 'rbf';
F16_RBF.trainFunct{1,1} = 'radbas';
F16_RBF.trainFunct{2,1} = 'purelin';


%% Levenberg-Marquardt algorithm
% Setting up the algorithm

w_init_output = ones(1,hidden_layer_size)*randn(1);
alpha_w       = randn(1)*ones(hidden_layer_size,1);
beta_w        = randn(1)*ones(hidden_layer_size,1);
V_w           = randn(1)*ones(hidden_layer_size,1);
w_init_in     = cat(2,alpha_w,beta_w,V_w);
a             = rand(hidden_layer_size,1);

% Get centers by Kmeans clustering
[~,centers]     = kmeans(Z_kt, hidden_layer_size,'MaxIter',10000);

F16_RBF.IW      = w_init_in; 
F16_RBF.LW      = w_init_output;
F16_RBF.a       = a;
F16_RBF.centers = centers;

% Get error of current network for first iteration
yRBF      = simNet(F16_RBF,Z_kt');
eold      = Cmt' -  yRBF.Y2;
wold      = cat(2,w_init_in,w_init_output',centers);
E         = mse(eold);
errorlist = 100*ones(F16_RBF.epochs,1);

for epoch = 1:F16_RBF.epochs
        
    % Construct Jacobian
    J = jacobian_rbf(wold,eold,Z_kt',yRBF,F16_RBF);

    % Calc new W
    wold = reshape(wold,1,7*hidden_layer_size);
    wnew = wold - (pinv(J'*J+ F16_RBF.mu*eye(size(J'*J)))*(J'*eold'))';

    wnew = reshape(wnew,hidden_layer_size,7);
    wold = reshape(wold,hidden_layer_size,7);
    
    % Replace weights in network with new weights
    F16_RBF.IW = wnew(:,1:3);
    F16_RBF.LW = wnew(:,4)';
    F16_RBF.centers = wnew(:,5:7);
    
    yRBF    = simNet(F16_RBF,Z_kt');
    enew    = Cmt'- yRBF.Y2;
    E1 = mse(enew);
    
    % If new error is smaller, adapt changes, increase mu
    if E > E1 && F16_RBF.mu < mu_max
        F16_RBF.mu = F16_RBF.mu*adapt_rate;
        wold = wnew;
        eold = enew;
        E = E1;
    % If new error is bigger, decrease mu
    elseif E <= E1 && F16_RBF.mu >= mu_min
        F16_RBF.mu      = F16_RBF.mu/adapt_rate;
        F16_RBF.IW      = wold(:,1:3);
        F16_RBF.LW      = wold(:,4)';
        F16_RBF.centers = wold(:,5:7);
        yRBF            = simNet(F16_RBF,Z_kt');
    end
    
    E1 = mse(enew);
    errorlist(epoch) = E1;
    e_grad = gradient(errorlist(1:epoch));
    
    % Check the stopping conditions
    if F16_RBF.mu >= mu_max 
        disp('Max mu reached')
        break
    elseif F16_RBF.mu <= mu_min
        disp('Min mu reached')
        break
    elseif mse(eold) == F16_RBF.goal
        disp('Goal reached')
        break
    elseif (epoch > 2) && (abs(e_grad(end)) <= F16_RBF.min_grad)
        disp('Minimum gradient reached')
        break
    end
end
its = epoch;
e.train = Ytrain' - yRBF.Y2;
yRBF_test = simNet(F16_RBF,Xtest');
e.test = Ytest' - yRBF_test.Y2;
disp(mse(e.test))
end


