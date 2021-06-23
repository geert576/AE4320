%% Preprocessing the data

function [Xtrain,Xtest,Ytrain,Ytest] = preprocess(X,Y,fraction,norm)
    if norm == true
        alpha = normalize(X(:,1));
        beta  = normalize(X(:,2));
        V     = normalize(X(:,3));
        X = cat(2,alpha,beta,V);
    end
    [Xtrain,Xtest,Ytrain,Ytest] = split_data(X,Y,fraction);

end

function [Xtrain,Xtest,Ytrain,Ytest] = split_data(X,Y,fraction)
    idx = length(Y)*fraction;
    Xtrain = X(1:idx,:);
    Xtest  = X(idx:end,:);
    Ytrain = Y(1:idx);
    Ytest  = Y(idx:end);
end