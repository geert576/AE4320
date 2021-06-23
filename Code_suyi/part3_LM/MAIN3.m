clc;
close all;
clear;
% Load the reconstructed data (alpha, beta, V) and the output data Cm
load('F16_reconstructed', 'X', 'Y');
rng('default');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Training and Testing Dataset + Initialize Cross Validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training data size
p_train = 0.9;

% Split the full data into training and testing dataset
[X_train, Y_train, X_test, Y_test] =split_data(p_train, X,Y);

% Save as a struct
data = struct('X', X, 'X_train', X_train, 'X_test', X_test, 'Y', Y, 'Y_train', Y_train, 'Y_test', Y_test,'preprocess',double.empty,'postprocess',double.empty);

% Preprocess (normalize) training data
[data, X_train_n, Y_train_n] = preprocess(data,X_train,Y_train);


%% Levenberg-Marquardt (LM) learning algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the Network Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up training algorithm
trainAlg = 'trainLinearRegression';

% Number of input neurons
n_input    = size(X,2);
% Number of output neurons
n_output   = size(Y,2);
% Number of hidden neurons
n_hidden        =20;

% Training parameters
epochs     = 1000;    % maximum number of epochs during training
goal       = 0;       %stops when goal is reached 
min_grad   = 1e-7;   %minimal gradient, training stops when absolute gradient
                      %dropsbelow this value
mu         = 1;       %learning rate, adapted during training
mu_dec     = 0.1;     %multiplication factor to decrease learning rate
mu_inc     = 10;      %multiplication factor to increase learning rate
mu_max     = 1e30;    %maximum learning factor

% Use k-means clustering method for initial neuron placement
[~,centers]     = kmeans(X_train_n, n_hidden,'MaxIter',10000);

% Center weights
netRBF.centers    = centers; 

% Network name
netRBF.name{1,1}  = 'rbf';

% Training algorithm
netRBF.trainAlg{2,1}         = trainAlg;

% Training parameters
netRBF.trainParam.epochs     = epochs;
netRBF.trainParam.goal       = goal;
netRBF.trainParam.min_grad   = min_grad;
netRBF.trainParam.mu         = mu;  
netRBF.trainParam.mu_dec     = mu_dec;
netRBF.trainParam.mu_inc     = mu_inc;
netRBF.trainParam.mu_max     = mu_max;

% Activation functions
netRBF.trainFunc{1,1}  = 'radbas';  %for the hidden layer
netRBF.trainFunc{2,1}  = 'purelin'; %for the output layer

% Bounds on the input space
netRBF.range = [-ones(n_input, 1) ones(n_input, 1)];

% Training algorithm
trainAlg = 'trainlm';
netRBF.trainAlg{2,1}= trainAlg;

% Number of center weights
netRBF.N_centers = [size(centers,1),1];

% Center weights
netRBF.centers  = centers; 

% Total number of çweights
n_weights = n_hidden * (n_input*2 + n_output);
%n_weights = n_hidden * (n_input + n_output);

% Random initializations of input and output weights
IW         = randn(n_hidden,n_input);
LW         = randn(n_output,n_hidden);

% Initialize array for cost function
E          = zeros(epochs,1);
mus        = zeros(epochs,1);
mu_min     = 1e-10;

for epoch = 1:epochs
    % Obtain the final normalized outputs (struct including the hidden layer outputs and
    % vj values)
    phi_j      = calc_phi_j(IW,X_train_n,centers);
    Y_hat      = calc_outputs(LW,phi_j);

    % Compute the Jacobian
    J=compute_jacobian(X_train_n, phi_j, IW, LW, centers);

    % Compute current cost function value
    [Et,e] = calc_cost(Y_train_n, Y_hat');
    E(epoch) = Et;
    mus(epoch)=mu;
    
    % Flatten the weights: 1 by n_weights row vector 
    Wt = reshape([IW LW' centers], 1, n_weights);
    %Wt = reshape([IW LW'], 1, n_weights);
    
    % Update the weights
    Wt1=Wt - ((J'*J+ mu*eye(size(J'*J)))\(J'*e))';
    
    % Unflatten the weights
%     Wt1_reshape = reshape(Wt1, n_hidden, n_input + n_output);
%     IW_new      = Wt1_reshape(:, 1:n_input);
%     LW_new      = Wt1_reshape(:, end)';
    Wt1_reshape = reshape(Wt1, n_hidden, n_input*2 + n_output);
    IW_new      = Wt1_reshape(:, 1:n_input);
    LW_new      = Wt1_reshape(:, n_input+1)';
    centers_new = Wt1_reshape(:, n_input+2:end);

    % Calculate the new outputs
    phi_j       = calc_phi_j(IW_new,X_train_n,centers);
    Y_hat       = calc_outputs(LW_new,phi_j);

    % Compute new cost function value
    [Et1,e] = calc_cost(Y_train_n, Y_hat');
    
    stop=1;
    
    % Compute gradients
    E_grad = gradient(E(1:epoch));
    if (epoch > 2) && (abs(E_grad(end)) <= min_grad)
        fprintf('Partial derivatives are (close to) zero \n');
    elseif epoch == epochs
        fprintf('Maximum number of epochs reached \n');
    elseif E(epoch) == goal
        fprintf('Goal reached \n');
    elseif (mu_min>mu)
        fprintf('Minimum learning rate reached \n');
    elseif (mu>mu_max)
        fprintf('Maximum learning rate reached \n');
    else
        stop = 0;
    end
    
    if stop ==1
        disp(epoch)
        break
    end
    
    % if new error smaller than old error
    if (Et1 < Et) && (mu_max>mu)

            % Accept the changes
            IW = IW_new;
            LW = LW_new;
            centers = centers_new;
            
            % Increase learning rate
            mu = mu * mu_inc;
            
    % if new cost function value is larger, do not accepte changes
    else
            % decrease learning rate
            mu = mu * mu_dec;
    end
    
end

% Save the input and output weights
netRBF.IW = IW;
netRBF.LW = LW;
netRBF.centers = centers;
netRBF.trainParam.mu         = mu;

% Simulate the final LM-RBF net
Y_hat_train_lm = simNet(netRBF, X_train_n');
% Reverse the transformation
Y_hat          = mapminmax('reverse',Y_hat_train_lm,data.postprocess);

% Plot results
fig_title      ='Levenberg-Marquardt Approach to RBFNN - IO mapping';
plot_3Dresults(X_train,Y_train,Y_hat',fig_title);

% net = feedforwardnet(3,'trainlm');
% [net,tr] = train(net,X_train_n',Y_train_n');
% y = net(X_train_n');
% y = mapminmax('reverse',y,data.postprocess);
% 
% %Plot results
% fig_title='Levenberg-Marquardt Approach to RBFNN - IO mapping';
% plot_3Dresults(X_train,Y_train,y',fig_title);
%% Helper functions

function [data,X_n,Y_n]=preprocess(data,X,Y)
% preprocess() transforms the input data (and output data for training) 
% in range [-1,1] and returns the process settings

% Inputs:
% - data: struct that consists the full dataset and split data
% - X: input data
% - Y: output data

% Outputs:
% - data: struct containing the new preprocessing setting (and 
% postprocessing setting)
% - X_n: normalized input data
% - Y_n: normalized output data if specified. 


% If no preprocessing has been done
if isempty(data.preprocess)
    [X_n, data.preprocess] = mapminmax(X'); 
% Apply existing preprocess settings to data
else 
    X_n =  mapminmax('apply',X', data.preprocess);
end

% Transpose the normalized data
X_n       = X_n';

% If preprocessing of output vector is wanted
if nargin == 3
    if isempty(data.postprocess)
        [Y_n, data.postprocess]= mapminmax(Y');        
    else 
        Y_n =  mapminmax('apply',Y', data.postprocess);
    end
    % Pass Y_n as optional output
    Y_n = Y_n';
end

end 

function phi_j=calc_phi_j(IW,X,centers)
n_data     = size(X,1);
n_input    = size(X,2);
n_hidden   = size(centers,1);
%Initialize array for storing input layer outputs
vj         = zeros(n_hidden,n_data);
% Calculate the outputs of the input layer
for i=1:n_input
   vj= vj+ (IW(:,i)*X(:,i)'-(IW(:,i).*centers(:,i))*ones(1,n_data)).^2;
end
vj         =vj';
% Calculate output of the hidden layer
phi_j  = exp(-vj);

end

function Y=calc_outputs(LW,phi_j)
%   Generating output for the output layer
Y  = LW*phi_j';
end

function J=compute_jacobian(X, phi_j, IW,LW, centers)
% compute_jacobian() calculates the Jacobian

% Inputs:
% -X: input vector consisting alpha,beta and V [#data points, #input variables]
% -phi-j: output of the hidden layers [#data points, #hidden neurons]
% -IW: input weights [#hidden neurons, #input neurons]
% -LW: output weights [#hidden neurons, #input neurons]
% -centers: RBF centers locations [#hidden neurons, #input neurons]

% Outputs:
% -J: Jacobian [#data points #weights]


% 1.Compute dependencies of errors wrt network outputs
de_dyk = -1;

% 2.Compute dependencies of yk wrt output layer input vk: purelin ->1
dyk_dvk = 1;

% 3.Comput dependencies of hidden layer outputs wrt hidden layer weights
% dvk_dwjk = yj
dvk_dwjk = phi_j;

% -> dependencies of errors wrt output weights de_dwjk
de_dwjk  = de_dyk * dyk_dvk * dvk_dwjk;

% 4.Compute dependencies of output layer inputs vk wrt hidden layer 
%activation function output yj
dvk_dyj = LW;

% 5.Compute dependencies of hidden layer activation function outputs yj (=phi_j(vj)=exp(-vj) wrt
% hidden layerp inputs vj
dyj_dvj = -phi_j;

% 6.Compute dependencies wrt input weights
% Derivatives of vj
n_data     = size(X,1);
n_input    = size(X,2); 
n_hidden   = size(centers,1);
dvj_dwij   = zeros(n_data,n_hidden,n_input);
de_dwij    = zeros(n_data,n_hidden,n_input);
for i=1:n_input
    % dependency of hidden layer input wrt input weights
    dvj_dwij(:,:,i) = 2 *IW(:,i)'.*((X(:,i)-centers(:,i)')).^2;
    % -> dependencies of errors wrt input weights de_dwij
    de_dwij(:,:,i)  = de_dyk* dyk_dvk * dvk_dyj .*dyj_dvj.* dvj_dwij(:,:,i);
end

% 7. Compute the dependçencies of error wrt to the center weights
de_dcij    = zeros(n_data,n_hidden,n_input);
for i=1:n_input
    dvj_dcij =  -2 *IW(:,i)'.^2.*(X(:,i)-centers(:,i)');
    de_dcij(:,:,i) = de_dyk* dyk_dvk * dvk_dyj .*dyj_dvj.* dvj_dcij;
end

% Construct the Jacobian [input weights, output weights, center weights]
J = [reshape(de_dwij, n_data, n_input*n_hidden) de_dwjk reshape(de_dcij, n_data, n_input*n_hidden)];
%J = [reshape(de_dwij, n_data, n_input*n_hidden) de_dwjk];
end

function [E,e] = calc_cost(Y,Y_hat)
% Calculate the error vector
e              = Y - Y_hat;

% Compute the cost function value = 0.5*MSE
E              = 0.5*sum(e.^2)/size(e,1);
end