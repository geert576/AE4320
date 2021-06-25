function [e,its,yRBF,F16_RBF,Z_k] = levenberg(mu,hidden_layer_size)
load('F16traindata_CMabV_2021.mat', 'Cm', 'Z_k', 'U_k');
F16_RBF = [];

mu_max = 5e30;
mu_min = 5e-10;
min_grad = 1e-17;
adapt_rate = 10;
F16_RBF.epochs = 1000;

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
% First iteration:

w_init_output = ones(1,hidden_layer_size)*randn(1);
alpha_w    = randn(1)*ones(hidden_layer_size,1);
beta_w     = randn(1)*ones(hidden_layer_size,1);
V_w        = randn(1)*ones(hidden_layer_size,1);
w_init_in  = cat(2,alpha_w,beta_w,V_w);
F16_RBF.IW = w_init_in; 
F16_RBF.LW = w_init_output;
a = rand(hidden_layer_size,1); 
F16_RBF.a = a;

% Get centers by Kmeans clustering
[~,centers]     = kmeans(Z_kt, hidden_layer_size,'MaxIter',10000);
F16_RBF.centers = centers;

yRBF    = simNet(F16_RBF,Z_kt');
eold      = Cmt' -  yRBF.Y2;
wold = cat(2,w_init_in,w_init_output',centers);
E = 0.5*sum(eold.^2)/size(eold,2);
errorlist = 100*ones(F16_RBF.epochs,1);

for epoch = 1:F16_RBF.epochs
        
    % Construct Jacobian
    J = jacobian_w(wold,eold,Z_kt',yRBF,F16_RBF);

   % calc new W
    wold = reshape(wold,1,7*hidden_layer_size);

    wnew = wold - (pinv(J'*J+ mu*eye(size(J'*J)))*(J'*eold'))';

    wnew = reshape(wnew,hidden_layer_size,7);
    wold = reshape(wold,hidden_layer_size,7);
    F16_RBF.IW = wnew(:,1:3);
    F16_RBF.LW = wnew(:,4)';
    F16_RBF.centers = wnew(:,5:7);
    
    yRBF    = simNet(F16_RBF,Z_kt');
    enew    =  Cmt'- yRBF.Y2;
    E1 = 0.5*sum(enew.^2)/size(enew,2);
    % disp(E1);
    
    if E > E1 && mu < mu_max
        mu = mu*adapt_rate;
        wold = wnew;
        eold = enew;
        E = E1;

        
        
    elseif E <= E1 && mu >= mu_min
        mu = mu/adapt_rate;
        F16_RBF.IW = wold(:,1:3);
        F16_RBF.LW = wold(:,4)';
        F16_RBF.centers = wold(:,5:7);
        yRBF    = simNet(F16_RBF,Z_kt');
    end
    E1 = size(enew,2)\0.5*sum(enew.^2);
    errorlist(epoch) = E1;
    e_grad = gradient(errorlist(1:epoch));
    if mu >= mu_max 
        disp('Max mu reached')
        break
    elseif mu <= mu_min
        disp('Min mu reached')
        break
    elseif mse(eold) == 0
        disp('Goal reached')
        break
    elseif (epoch > 2) && (abs(e_grad(end)) <= min_grad)
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


