%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Geert KF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;
randn('seed', 7);
load('F16traindata_CMabV_2021.mat', 'Cm', 'Z_k', 'U_k');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n           = 4;        % number of states
nm          = 3;        % number of measurements
m           = 3;        % number of inputs
dt          = 0.01;     % time step (s)
N           = length(Cm);     % sample size
T           = linspace(0,dt*N,N);
epsilon         = 1e-10;
doIEKF          = 1;        % set 1 for IEKF and 0 for EKF
maxIterations   = 100;

printfigs   = 0;
figpath     = '';

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values for states and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E_x_0       = [150 ;
               0 ;
               0 ;
               0]; % initial estimate of optimal value of x_k1_k1
% x_0         = 5;        % initial true state

% B           = 1;        % input matrix
G           = 1;        % noise input matrix

% Initial estimate for covariance matrix
std_x_0     = [5 0 0 0;
               0 5 0 0;
               0 0 5 0;
               0 0 0 5];
P_0         = std_x_0^2;

% System noise statistics:
E_w         = 0;                            % bias of system noise
std_w       = [1e-3 0 0 0;
               0 1e-3 0 0;
               0 0 1e-3 0;
               0 0 0    0];                 % standard deviation of system noise
Q           = std_w^2;                      % variance of system noise
% w_k         = std_w * randn(n, N) + E_w;    % system noise

% Measurement noise statistics:
E_v         = 0;                            % bias of measurement noise
std_v       = [0.035 0 0;
               0 0.013 0;
               0 0  0.11];% standard deviation of measurement noise
R           = std_v^2;                      % variance of measurement noise
% v_k         = std_v * randn(n, N) + E_v;    % measurement noise

% for numerical demo only
% w_k(1)      = -2;
% v_k(1)      = 1;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate batch with measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% True state and measurements data:
% x           = x_0;              % initial true state
% X_k         = zeros(n, N);      % true state
% Z_k         = zeros(nm, N);     % measurement
% U_k         = zeros(m, N);      % inputs
% for i = 1:N
%     dx          = kf_calc_f(0, x, U_k(:,i));            % calculate noiseless state derivative     
%     x           = x + (dx + w_k(:,i))*dt;               % calculate true state including noise
%     X_k(:,i)    = x;                                    % store true state
%     Z_k(:,i)    = kf_calc_h(0, x, U_k(:,i)) + v_k(:,i); % calculate and store measurement 
% end

% Get true state from noise free measurements
u_true = cumtrapz(T,U_k(:,1));
v_true = cumtrapz(T,U_k(:,2));
w_true = cumtrapz(T,U_k(:,3));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_k             = 0; 
t_k1            = dt;

XX_k1_k1        = zeros(n, N);
PP_k1_k1        = zeros(n,n, N);
STD_x_cor       = zeros(n, N);
STD_z           = zeros(nm, N);
ZZ_pred         = zeros(nm, N);

x_k1_k1         = E_x_0;    % x(0|0)=E{x_0}
P_k1_k1         = P_0;      % P(0|0)=P(0)

B               = kf_calc_lin_B();
G               = [1 0 0 0;
                   0 1 0 0;
                   0 0 1 0;
                   0 0 0 0];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; % start timer

% Run the filter through all N samples
for k = 1:N
    % x(k+1|k) (prediction)
    [t, x_k1_k]     = rk4(@kf_calc_f, x_k1_k1, U_k(k,:), [t_k, t_k1]); 
    
    Fx = zeros(4);               
    % the continuous to discrete time transformation of Df(x,u,t)
    [dummy, Psi]    = c2d(Fx, B, dt);   
    [Phi, Gamma]    = c2d(Fx, G, dt);   
    
    % x(k+1|k) (prediction)
    % x_k1_k      = Phi*x_k1_k1 + Psi*U_k(k,:)';
    
    % Observation and observation error predictions
    z_k1_k      = kf_calc_h(0, x_k1_k, U_k(k,:));
            
    % P(k+1|k) (prediction covariance matrix)
    P_k1_k          = Phi*P_k1_k1*Phi.' + Gamma*Q*Gamma.';  
    % Run the Iterated Extended Kalman filter (if doIEKF = 1), else run standard EKF
    if (doIEKF)
        % do the iterative part
        eta2    = x_k1_k;
        err     = 2*epsilon;

        itts    = 0;
        while (err > epsilon)
            if (itts >= maxIterations)
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
                break
            end
            itts    = itts + 1;
            eta1    = eta2;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx       = kf_calc_Hx(0, eta1, U_k(k,:)); 
            
            % Check observability of state
            if (k == 1 && itts == 1)
                rankHF  = kf_calcObsRank(Hx, Fx);
                if (rankHF < n)
                    warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
                end
            end
         
            % Observation and observation error predictions
            z_k1_k      = kf_calc_h(0, x_k1_k, U_k(k,:));     % prediction of observation 
            P_zz        = (Hx*P_k1_k*Hx.' + R);             % covariance matrix of observation error
            std_z       = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation) 

            % calculate the Kalman gain matrix
            K           = P_k1_k * Hx.'/P_zz;
            % new observation state
            z_p         = kf_calc_h(0, eta1, U_k(k,:));

            eta2        = x_k1_k + K*(Z_k(k,:)' - z_p - Hx*(x_k1_k - eta1));
            err         = norm((eta2 - eta1), inf)/norm(eta1, inf);
        end

        IEKFitcount(k)  = itts;
        x_k1_k1         = eta2;
    else
        % Correction
        Hx              = kf_calc_Hx(0, x_k1_k, U_k(k,:));  % perturbation of h(x,u,t)
        
        % P_zz(k+1|k) (covariance matrix of innovation)  
        P_zz            = (Hx*P_k1_k * Hx.' + R);           % covariance matrix of observation error
        std_z           = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation)    

        % K(k+1) (gain)
        K               = P_k1_k * Hx.'/P_zz;
        
        % Calculate optimal state x(k+1|k+1) 
        x_k1_k1         = x_k1_k + K*(Z_k(k,:)' - z_k1_k); 
    end    
    % Correction
    % Hx              = kf_calc_Hx(0, x_k1_k, U_k(k,:)); % perturbation of h(x,u,t)
    
    % Observation and observation error predictions
    % z_k1_k          = kf_calc_h(0, x_k1_k, U_k(k,:));     % prediction of observation 
    % P_zz            = (Hx*P_k1_k * Hx.' + R);           % covariance matrix of observation error
    % std_z           = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation)        

    % K(k+1) (gain)
    % K               = P_k1_k * Hx.'/P_zz;
    
    % Calculate optimal state x(k+1|k+1) 
    % x_k1_k1         = x_k1_k + K * (Z_k(k,:)' - z_k1_k); 

    % P(k+1|k+1) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1; 
    P_k1_k1         = (eye(n) - K*Hx) * P_k1_k * (eye(n) - K*Hx).' + K*R*K.';  
    std_x_cor       = sqrt(diag(P_k1_k1));             % standard deviation of state estimation error (for validation)

    % Next step
    t_k             = t_k1; 
    t_k1            = t_k1 + dt;
    
    % store results
    XX_k1_k1(:,k)     = x_k1_k1;
    PP_k1_k1(:,:,k)   = P_k1_k1;
    STD_x_cor(:,k)    = std_x_cor;
    STD_z(:,k)        = std_z;
    ZZ_pred(:,k)      = z_k1_k;
end

time = toc; % end timer

% calculate state estimation error (in real life this is unknown!)
% EstErr_x    = XX_k1_k1-X_k;

% calculate measurement estimation error (possible in real life)
EstErr_z    = ZZ_pred-Z_k';

% fprintf('EKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr_x)), N, time);

save('z_kalman.mat','ZZ_pred','XX_k1_k1','Cm','STD_z','STD_x_cor');
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
plot(u_true, 'b');
plot(Z_k(:,1), 'k');
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
subplot(3,1,1);
plot(T,u_true, 'b',T,XX_k1_k1(1,:), 'r');
title('True state and estimated state (u)');
subplot(3,1,2);
plot(T,v_true, 'b',T,XX_k1_k1(2,:), 'r');
title('True state and estimated state (v)');
subplot(3,1,3);
plot(T,w_true, 'b',T,XX_k1_k1(3,:), 'r');
title('True state and estimated state (w)');
legend('True state', 'Estimated state', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateEstimatedState";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

a_m = ZZ_pred(1,:);
c_a = XX_k1_k1(4,:);

a_true = a_m./(1+c_a);

plotID = 1003;
figure(plotID);
% set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
% plot(X_k, 'b');
subplot(3,1,1);
plot(T,Z_k(:,1), 'k.',T,ZZ_pred(1,:), 'r',T,a_true,'b');
subplot(3,1,2);
plot(T,Z_k(:,2), 'k.',T,ZZ_pred(2,:), 'r');
subplot(3,1,3);
plot(T,Z_k(:,3), 'k.',T,ZZ_pred(3,:), 'r');
title('True state, estimated state and measurement');
legend( 'Measurement', 'Estimated state', 'Location', 'northeast');
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
% plot(EstErr_x, 'b');
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
set(plotID, 'Position', [600 0 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
% plot(EstErr_x, 'b');
plot(STD_x_cor, 'r');
plot(-STD_x_cor, 'g');
% axis([0 50 min(EstErr_x) max(EstErr_x)]);
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
set(plotID, 'Position', [600 0 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
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
