%% Calculate the jacobian for the levenberg algorithm

function J = jacobian_rbf(w,e,x,output,net)
w_in    = w(:,1:3);
w_out   = w(:,4);
centers = w(:,5:7);

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

dyjdvj = -output.Y1;

% Dependencies wrt input weights
% dE/dwij
% TODO iterate over inputs to get 3 columns of weights in
neurons = size(w_in,1);
for i = 1:size(w_in,2)
    % Input weights
    dvdwij = 2*w_in(:,i).*(x(i,:)-centers(:,i)).^2;
    J(:,((i-1)*neurons+1):i*neurons) = (dEdyk * dykdvk .* dvkdyj .* dyjdvj .* dvdwij)';
    %J(:,((i-1)*neurons+1):i*neurons) = (dEdyk .* (dykdvk * dvkdyj .* dyjdvj .* dvdwij))';
    %J(:,((i-1)*neurons+1):i*neurons) = dEdyk * dykdvk * output.Y1' .* dyjdvj .* dvdwij;
    
    % Centers
    dvjcij = -2*(w_in(:,i).^2).*(x(i,:)-centers(:,i));
    J(:,((i+3)*neurons+1):(i+4)*neurons) = (dEdyk * dykdvk .* dvkdyj .* dyjdvj .* dvjcij)';
end

% Output weights
J(:,3*neurons+1:4*neurons) = dEdwjk';


end
