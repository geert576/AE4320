function [X_train, Y_train, X_test, Y_test] =split_data(p_train, X,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% split_data splits the full data into training and testing dataset 

% Inputs:
% - p_train: percentage of training data over the full dataset
% - X:       input matrix    
% - Y:       output matrix 

% Outputs:
% - X_train, Y_train, X_test, Y_test: training and testing input and output
% matrices. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obtain the size of the data samples
N = size(X,1);

%Shuffle the dataset indices randomly
idx = randperm(N);

% Set training idx
train_idx = round(p_train*N);

% Get the training and testing data
X_train = X(idx(1:train_idx), :);
Y_train = Y(idx(1:train_idx), :);

X_test = X(idx(train_idx+1:end), :);
Y_test = Y(idx(train_idx+1:end), :);

end