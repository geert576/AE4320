%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H = kf_calcDHx(x) Calculates the Jacobian of the output dynamics equation f(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hx = kf_calc_Hx(t, x, u)

    u = x(1);
    v = x(2);
    w = x(3);
    Caup = x(4);
    % Calculate Jacobian matrix of output dynamics
    Hx = [(-w/(u^2+w^2)) * (1 + Caup) , 0,  (u/(u^2+w^2)) * (1 + Caup), atan(w/u);
          -(v*u)/(((w^2+u^2)^0.5)*(v^2+w^2+u^2))  , ((w^2+u^2)^0.5)/(v^2+w^2+u^2), -(v*w)/(((w^2+u^2)^0.5)*(v^2+w^2+u^2)), 0;
          u/(v^2+w^2+u^2)^0.5 ,v/(v^2+w^2+u^2)^0.5 ,w/(v^2+w^2+u^2)^0.5 ,0 ];
    
               
    end
