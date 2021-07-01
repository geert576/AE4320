%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% zpred = kf_calcHx(x) Calculates the output dynamics equation h(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zpred = kf_calcHx(t, x, u)
    u = x(1);
    v = x(2);
    w = x(3);
    Ca = x(4);
    zpred = [atan(w/u)*(1+Ca);
             atan(v/sqrt(u^2+w^2));
             sqrt(u^2 + v^2 + w^2)];
    end
    