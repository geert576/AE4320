%% Jacobian for the feedforward network

function J = jacobian_ff(w,e,x,output,net)
w_in    = w(:,1:3);
w_out   = w(:,4);


J = zeros(numel(e),numel(w));

% Dependencies wrt network outputs
% dE/dyk

dEdyk = -1;%e;

% Dependencies wrt output layer input vk
% dyk/dvk

dykdvk = 1; % linear activation function

% Dependencies wrt hidden layer weights
% dE/dwjk

dEdwjk = dEdyk .* output.Y1;

% Dependencies of hidden layer activation function yj 
% dE/dyj

%dEdyj = dEdyk * dykdvk * w_out;

dvkdyj = w_out;
% Dependencies of hidden layer activation function yj wrt to vj
% dyj/dvj

dyjdvj = (4*exp(-2*output.V1))./(1+exp(-2*output.V1)).^2;

% Dependencies wrt input weights
% dE/dwij
% TODO iterate over inputs to get 3 columns of weights in
neurons = size(w_in,1);

for i = 1:size(w_in,2)
    % Input weights
    dvjdwij    = x(:,i);
    J(:,((i-1)*neurons+1):i*neurons) = (dEdyk*dykdvk.*dvkdyj.*dyjdvj.*dvjdwij')';
    %J(:,((i-1)*neurons+1):i*neurons) = (dEdyk .* (dykdvk * dvkdyj .* dyjdvj .* dvdwij))';
    %J(:,((i-1)*neurons+1):i*neurons) = dEdyk * dykdvk * output.Y1' .* dyjdvj .* dvdwij;
end

% Output weights
J(:,3*neurons+1:4*neurons) = dEdwjk';


end