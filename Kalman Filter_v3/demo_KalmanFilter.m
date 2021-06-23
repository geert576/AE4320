%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the Linear Kalman Filter for linear systems
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;
randn('seed', 7);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n           = 1;        % number of states
dt          = 0.01;     % time step (s)
N           = 1000;     % sample size

printfigs   = 0;
figpath     = '';

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values for states and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E_x_0       = 10;       % initial estimate of optimal value of x_k_1k_1
x_0         = 1;        % initial true state
u           = 0;        % system input

B           = 0;        % input matrix
G           = 1;        % noise input matrix

% Initial estimate for covariance matrix
std_x_0     = 10;                       % initial standard deviation of state prediction error
P_0         = std_x_0^2;                % initial covariance of state prediction error

% System noise statistics
E_w         = 0;                        % bias of system noise
std_w       = 100;                      % standard deviation of system noise
Q           = std_w^2;                  % variance of system noise
w_k         = std_w*randn(N, n) + E_w;  % system noise

% Measurement noise statistics
E_v         = 0;                        % bias of measurement noise
std_v       = 2;                        % standard deviation of measurement noise 
R           = std_v^2;                  % variance of measurement noise
v_k         = std_v*randn(N, n) + E_v;  % measurement noise

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate batch with measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Real simulated state-variable and measurements data:
x       = x_0;              % initial true state
X_k     = zeros(N, 1);      % true state
Z_k     = zeros(N, 1);      % measurement
U_k     = zeros(N, 1);      % inputs
for i = 1:N
    dx          = kf_calc_lin_F(0, x, U_k(i));              % calculate noiseless state derivative
    x           = x + (dx + w_k(i,:))*dt;                   % calculate true state including noise
    X_k(i,:)    = x;                                        % store true state
    Z_k(i,:)    = kf_calc_lin_H(0, x, U_k(i)) + v_k(i,:);   % calculate and store measurement 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_k             = 0; 
t_k1            = dt;

% Initialize matrices for storing data
XX_k1_k1        = zeros(N, n);
PP_k1_k1        = zeros(N, n);
STD_x_cor       = zeros(N, n);
STD_z           = zeros(N, n);
ZZ_pred         = zeros(N, n);

% initialize state estimate and error covariance
x_k1_k1         = E_x_0;     % x(0|0)=E{x_0}
P_k1_k1         = P_0;       % P(0|0)=P(0)

% Calculate discrete state transition matrix and prediction (Time and State invariant!)
F               = kf_calc_lin_F(0, 1, u);
[dummy, Psi]    = c2d(F, B, dt);   
[Phi, Gamma]    = c2d(F, G, dt);   
% Calculate discrete state observation matrix (Time invariant!)
H               = kf_calc_lin_H(0, 1, 0);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; % start timer 

% Run the filter through all N samples
for k = 1:N   
    % x(k+1|k) (prediction)
    x_k1_k      = Phi*x_k1_k1 + Psi*U_k(k);

    % P(k+1|k) (prediction covariance matrix)
    P_k1_k      = Phi*P_k1_k1*Phi.' + Gamma*Q*Gamma.'; 
    
    % Observation and observation error predictions
    z_k1_k      = H*x_k1_k;                 % prediction of observation (for validation)   
    P_zz        = H*P_k1_k*H' + R;          % covariance matrix of observation error (for validation)   
    std_z       = sqrt(diag(P_zz));         % standard deviation of observation error (for validation)        

    % K(k+1) (gain)
    K           = P_k1_k*H.'/(H*P_k1_k*H.' + R);   
    
    % Calculate optimal state x(k+1|k+1) 
    x_k1_k1     = x_k1_k + K*(Z_k(k) - H*x_k1_k);       
    
    % P(k|k) (correction)
    P_k1_k1     = (eye(n) - K*H)*P_k1_k;    
    std_x_cor   = sqrt(diag(P_k1_k1));      % standard deviation of state estimation error (for validation)
    
    % Next step
    t_k         = t_k1; 
    t_k1        = t_k1 + dt;
    
    % store results
    XX_k1_k1(k)     = x_k1_k1;
    PP_k1_k1(k)     = P_k1_k1;
    STD_x_cor(k)    = std_x_cor;
    STD_z(k)        = std_z;
    ZZ_pred(k)      = z_k1_k;    
end

time        = toc; % end timer

% calculate state estimation error (in real life this is unknown!)
EstErr_x    = XX_k1_k1-X_k;

% calculate measurement estimation error (possible in real life)
EstErr_z    = ZZ_pred-Z_k;

fprintf('KF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr_x)), N, time);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

plotID  = 1001;
figure(plotID);
set(plotID, 'Position', [1 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(X_k, 'b');
plot(Z_k, 'k');
title('True state and measurement');
legend('True state', 'Measurement', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateMeasurement";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 1002;
figure(plotID);
set(plotID, 'Position', [1 0 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(X_k, 'b');
plot(XX_k1_k1, 'r');
title('True state and estimated state');
legend('True state', 'Estimated state', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateEstimatedState";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 1003;
figure(plotID);
set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(X_k, 'b');
plot(XX_k1_k1, 'r');
plot(Z_k, 'k');
title('True state, estimated state and measurement');
legend('True state', 'Estimated state', 'Measurement', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateEstimatedStateMeasurement";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

pause;

plotID = 2001;
figure(plotID);
set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(EstErr_x, 'b');
plot(STD_x_cor, 'r');
plot(-STD_x_cor, 'g');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
title('State estimation error with STD');
if (printfigs == 1)
    name       = "StateEstimationError";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 2002;
figure(plotID);
set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(EstErr_x, 'b');
plot(STD_x_cor, 'r');
plot(-STD_x_cor, 'g');
axis([0 50 min(EstErr_x) max(EstErr_x)]);
title('State estimation error with STD (zoomed in)');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
if (printfigs == 1)
    name       = "StateEstimationErrorZoom";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

pause;

plotID = 3001;
figure(plotID);
set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(EstErr_z, 'b');
plot(STD_z, 'r');
plot(-STD_z, 'g');
legend('Measurement estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
title('Measurement estimation error with STD');
if (printfigs == 1)
    name       = "MeasurementEstimationError";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 3002;
figure(plotID);
set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(EstErr_z, 'b');
plot(STD_z, 'r');
plot(-STD_z, 'g');
axis([0 50 min(EstErr_z) max(EstErr_z)]);
title('Measurement estimation error with STD (zoomed in)');
legend('Measurement estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
if (printfigs == 1)
    name       = "MeasurementEstimationErrorZoom";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end
