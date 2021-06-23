%%%%
% LS estimator between Cm and alpha from kalman filter
%%%%
clear all
close all
printfigs = 1;
load('z_kalman');

% Load the parameters
N = length(STD_x_cor(4,:));

a_m = ZZ_pred(1,:);
b_m = ZZ_pred(2,:);
c_a = XX_k1_k1(4,:);

a_true = a_m./(1+c_a);

poly_order = 3;


%% Least squares calculations
x = a_true';
y = b_m';
Z = Cm;

N = length(x);


if poly_order == 1
    X = [ones(size(x)) x y];
end
if poly_order == 2
    X = [ones(size(x)) x y x.^2 y.^2 x.*y];
end
if poly_order == 3
    X = [ones(size(x)) x y x.^2 y.^2 x.*y x.^3 y.^3 x.^2.*y y.^2.*x];
end
if poly_order == 4
    X = [ones(size(x)) x x.^2 x.^3 x.^4];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do ordinary least squares estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
COV = inv(X'*X);
chat = COV * X' * Z;

% Determine the residual and W matrix for the WLS
z_true = Cm;

z_approx_ols = zeros(length(a_true),1);
for i = 1:length(a_true)
    x = a_true(i);
    y = b_m(i);
    if poly_order == 1
        Xval = [ones(size(x)) x y];
    end
    if poly_order == 2
        Xval = [ones(size(x)) x y x.^2 y.^2 x.*y];
    end
    if poly_order == 3
        Xval = [ones(size(x)) x y x.^2 y.^2 x.*y x.^3 y.^3 x.^2.*y y.^2.*x];
    end
    z_approx_ols(i) = Xval * chat;
    
end
residuals_ols = z_true - z_approx_ols;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The weighted least squares calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W = calcW(a_true,residuals_ols);
W_inv = inv(W);
COV = inv(X'* W_inv *X);
chat = COV * X' * W_inv * Z;

x = a_true';
y = b_m';

dx = 0.01;
dy = 0.01;


xval = min(x):dx:max(x);
yval = min(y):dy:max(y);
Z = zeros(length(xval),length(yval));
i = 1;

for x = xval
    j = 1;
    for y = yval
        if poly_order == 1
            Xval = [ones(size(x)) x y];
        end
        if poly_order == 2
            Xval = [ones(size(x)) x y x.^2 y.^2 x.*y];
        end
        if poly_order == 3
            Xval = [ones(size(x)) x y x.^2 y.^2 x.*y x.^3 y.^3 x.^2.*y y.^2.*x];
        end
        Z(i,j) = Xval * chat ;
        j = j + 1;
    end
    i = i + 1;
end


%% Determine absolute error

xcoords = a_true;
ycoords = b_m;
z_true = Cm;

z_approx = zeros(length(a_true),1);
for i = 1:length(a_true)
    x = a_true(i);
    y = b_m(i);
    if poly_order == 1
        Xval = [ones(size(x)) x y];
    end
    if poly_order == 2
        Xval = [ones(size(x)) x y x.^2 y.^2 x.*y];
    end
    if poly_order == 3
        Xval = [ones(size(x)) x y x.^2 y.^2 x.*y x.^3 y.^3 x.^2.*y y.^2.*x];
    end
    z_approx(i) = Xval * chat;
    
end

residuals = z_true - z_approx;
error = mean((residuals).^2);
var_par = diag(inv(X'* W_inv *X));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1)
scatter3(b_m,a_true,Cm,5,'filled');
hold on
surf(yval,xval,Z,'FaceAlpha',0.5);
xlabel('\beta [rad]');
ylabel('\alpha [rad]');
zlabel('C_m(\beta,\alpha)[-]');
figure(2)
plot(a_true,residuals);

figure(3)
bar(var_par);

figure(4)
histogram(residuals);

