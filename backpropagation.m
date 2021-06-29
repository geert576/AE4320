function [dw_in,dw_out] = backpropagation(e,x,output,net)
    w_in  = net.IW;
    w_out = net.LW;
    % Dependencies wrt network outputs
    % dE/dyk

    dEdyk = -e;%1;

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
    
    input_n  = size(w_in,2);
    hidden_n = size(w_in,1);
    dw_in = zeros(size(w_in));
    for i = 1:input_n
        dvjdwij    = x(:,i);
        dw_in(:,i) = sum((dEdyk*dykdvk.*dvkdyj'.*dyjdvj.*dvjdwij')')';
    end
    
    dw_out = sum(dEdwjk');
    
end