function plot_3Dresults(X,Y,Y_hat,fig_title)
% creating triangulation (only needed for plotting)
TRIeval     = delaunayn(X(:,[1,2]),{'Qbb','Qc','QJ1e-6'});

%creating figure for RBF network
figure; hold on
%viewing angles
az = 160;
el = 50;

%set up figure title
titlestring = sprintf(fig_title);
if size(X,2)==3
    sgtitle(titlestring)
    subplot(2, 1, 1); hold on; 
    % alpha in x axis, beta in y axis 
    plot3(X(:, 1), X(:, 2), Y, '.k'); % note that alpha_m = x(:, 1), beta_m = x(:, 2), Y = Cm
    grid on;

    trisurf(TRIeval, X(:, 1), X(:, 2), Y_hat, 'EdgeColor', 'none'); 
    grid on;
    view(az, el);
    xlabel('$\alpha$ [rad]', 'Interpreter', 'Latex');
    ylabel('$\beta$ [rad]', 'Interpreter', 'Latex');
    zlabel('$C_m(\alpha, \beta, V) [-]$', 'Interpreter', 'Latex');
    legend({'True', 'Hypothesis'}, 'Location', 'northeast');

    subplot(2, 1, 2); hold on; 
    % alpha in x axis, V in y axis 
    plot3(X(:, 1), X(:, 3), Y, '.k'); % note that alpha_m = x(:, 1), beta_m = x(:, 2), Y = Cm
    grid on;

    trisurf(TRIeval, X(:, 1), X(:, 3), Y_hat, 'EdgeColor', 'none'); 
    grid on;
    view(az, el);
    xlabel('$\alpha$ [rad]', 'Interpreter', 'Latex');
    ylabel('$V$ [m/s]', 'Interpreter', 'Latex');
    zlabel('$C_m(\alpha, \beta, V) [-]$', 'Interpreter', 'Latex');
    legend({'True', 'Hypothesis'}, 'Location', 'northeast');
    position=[100 100 600 900];
else 
    sgtitle(titlestring)
    % alpha in x axis, beta in y axis 
    plot3(X(:, 1), X(:, 2), Y, '.k'); % note that alpha_m = x(:, 1), beta_m = x(:, 2), Y = Cm
    grid on;

    trisurf(TRIeval, X(:, 1), X(:, 2), Y_hat, 'EdgeColor', 'none'); 
    grid on;
    view(az, el);
    xlabel('$\alpha$ [rad]', 'Interpreter', 'Latex');
    ylabel('$\beta$ [rad]', 'Interpreter', 'Latex');
    zlabel('$C_m(\alpha, \beta) [-]$', 'Interpreter', 'Latex');
    legend({'True', 'Hypothesis'}, 'Location', 'northeast')
    position = [100 100 600 500];
end
%   ... set fancy options for plotting RBF network
set(gcf,'Renderer','OpenGL','Position', position);
hold on;
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong;
drawnow();
end