function Hx = kf_calcDHx(t, x, u)

    u = x(1);
    v = x(2);
    w = x(3);
    Caup = x(4);
    % Calculate Jacobian matrix of output dynamics
    Hx = [(-1/u^2)*(1/(1+(w/u)^2)) * (1 + Caup) , 0,  (1/(1+(w/u)^2) * (1 + Caup)), arctan(w/u);
          -(v*u)/((w^2+u^2)^0.5*(v^2+w^2+u^2))  , (w^2+u^2)^0.5/(v^2+w^2+u^2), -(v*w)/((w^2+u^2)^0.5*(v^2+w^2+u^2)), 0;
          u/(v^2+w^2+u^2)^0.5 ,v/(v^2+w^2+u^2)^0.5 ,w/(v^2+w^2+u^2)^0.5 ,0; ]
    
               
    end