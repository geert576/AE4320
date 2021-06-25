function [e,yRBF,F16_RBF,Z_k] = RBF_ls(neurons,plot_rbf)
load('F16traindata_CMabV_2021.mat', 'Cm', 'Z_k', 'U_k');

% Estimating the F16 training data with a RBF NN
F16_RBF = [];
[Xtrain,Xtest,Ytrain,Ytest] = preprocess(Z_k,Cm,0.9,true);
alpha = Xtrain(:,1);
beta  = Xtrain(:,2);
V     = Xtrain(:,3);
Z_kn  = Ytrain;

% Init
hidden_layer_size = neurons;
weight_input = rand(1);
weight_output = randn(1);
alpha_w = randn(1)*ones(hidden_layer_size,3);
beta_w  = randn(1)*ones(hidden_layer_size,3);
V_w     = randn(1)*ones(hidden_layer_size,3);
F16_RBF.IW = cat(2,alpha_w,beta_w,V_w);
F16_RBF.LW = ones(1,hidden_layer_size)*weight_output;
F16_RBF.range = ones(3,2);
F16_RBF.a = rand(1); % The rbf amplitude

F16_RBF.name{1,1} = 'rbf';
F16_RBF.trainFunct{1,1} = 'radbas';
F16_RBF.trainFunct{2,1} = 'purelin';

% Get centers by Kmeans clustering
[idx,centers_a] = kmeans(alpha,hidden_layer_size);
[idx,centers_b] = kmeans(beta ,hidden_layer_size);
[idx,centers_v] = kmeans(V    ,hidden_layer_size);

F16_RBF.centers = cat(2,centers_a,centers_b,centers_v); % center weights [# hidden neurons, # input neurons]

yRBF    = simNet(F16_RBF,Z_kn');

X = yRBF.LS';
Z = Cm;
COV = inv(X'*X);
chat = COV * X' * Z;
F16_RBF.a = chat;

yRBF = simNet(F16_RBF,Z_kn');

%% Plotting
x = alpha;
y = beta;
dx = 0.01;
dy = 0.01;

xval = min(x):dx:max(x);
yval = min(y):dy:max(y);
Z = zeros(length(xval),length(yval));
i = 1;

for x = xval
    j = 1;
    [d,ix] = min(abs(alpha-x));
    for y = yval
        yRBF = simNet(F16_RBF,[x; y; V(ix)]);     
        Z(i,j) = yRBF.Y2;
        j = j + 1;
    end
    i = i + 1;
end

if plot_rbf == true
    figure(1)
    scatter3(beta,alpha,Cm,5,'filled');
    hold on
    surf(yval,xval,Z,'FaceAlpha',0.5);
    xlabel('\beta [rad]');
    ylabel('\alpha [rad]');
    zlabel('C_m(\beta,\alpha)[-]');

    plotting(F16_RBF,Z_kn);
end

e.train = Ytrain - yRBF.Y2;

yRBF    = simNet(F16_RBF,Xtest');
e.test  = Ytest - yRBF.Y2;
end