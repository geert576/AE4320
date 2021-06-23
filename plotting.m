%% The plotting script

function plotting(net,data)
    load('F16traindata_CMabV_2021.mat', 'Cm', 'Z_k', 'U_k');
    %%   Creating IO data points
    %---------------------------------------------------------
    
    alpha       = data(:,1);
    beta        = data(:,2);
    V           = data(:,3);
    minXI       = min(alpha)*ones(1,2);
    maxXI       = max(alpha)*ones(1,2);
    minYI       = min(beta)*ones(1,2);
    maxYI       = max(beta)*ones(1,2);
    N = 100;
    dx = (maxXI(1)-minXI(1))/N;
    dy = (maxYI(1)-minYI(1))/N;
    dv = (max(V)-min(V))/N;

    [xxe yye]   = ndgrid((minXI(1):dx:maxXI(1))', (minYI(2):dy:maxYI(2)'));
    [yyv vvv]   = ndgrid((minYI(1):dy:maxYI(1))', (min(V):dv:max(V)'));
    vve = mean(V)*ones(size(xxe));
    xxv = mean(alpha)*ones(size(vvv));
    Xevalx       = [xxe(:), yye(:), vve(:)]';
    Xevalv       = [xxv(:), yyv(:), vvv(:)]';
    delaunx = [xxe(:) yye(:)]';
    delaunv = [yyv(:) vvv(:)]';
    %delaun_unique = unique(delaun,'row');
    %%   Simulating the network output
    %---------------------------------------------------------
    %yFF     = simNet(netFF,Xeval);
    yRBFx    = simNet(net,Xevalx);
    yRBFv    = simNet(net,Xevalv);
    %%   Plotting results
    %---------------------------------------------------------
    %   ... creating triangulation (only needed for plotting)
    TRIevalx     = delaunayn(delaunx',{'Qbb','Qc','QJ1e-6'});
    TRIevalv     = delaunayn(delaunv',{'Qbb','Qc','QJ1e-6'});

    %   ... viewing angles
    az = 140;
    el = 36;

    %   ... creating figure for FF network
    plotID = 1012;
    figure(plotID);
    scatter3(beta,alpha,Cm,5,'filled');
    hold on
    trisurf(TRIevalx,  Xevalx(2, :)',Xevalx(1, :)', yRBFx.Y2', 'EdgeColor', 'none'); 
    grid on;
    %view(az, el);
    %titstring = sprintf('Feedforward neural network - IO mapping');
    xlabel('x_1');
    xlabel('x_2');
    zlabel('y');

    %   ... creating figure for FF network
%     plotID = 1013;
%     figure(plotID);
%     scatter3(beta,alpha,Cm,5,'filled');
%     hold on
%     trisurf(TRIevalv,  Xevalv(1, :)',Xevalv(2, :)', yRBFv.Y2', 'EdgeColor', 'none'); 
%     grid on;
%     %view(az, el);
%     %titstring = sprintf('Feedforward neural network - IO mapping');
%     xlabel('x_1');
%     xlabel('x_2');
%     zlabel('y');


end
