%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the symbolic construction of the state observation matrix
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear all;

% define variables
syms('Vx', 'Vz', 'Ax', 'Lx', 'Az', 'Lz', 'g', 'th', 'q', 'Lq', 'ze');

% define state vector
x  = [ Vx;  Vz;  th;  ze  ];
x0 = [ 100; 20;  .01; 1240]; % initial state

% define augmented state vector
xaug  = [x;   Lx; Lz; Lq];
xaug0 = [x0;  1;  1;  1 ]; % initial augmented state

% define parameter vector
p  = [ g    ];
p0 = [ 9.81 ]; % "known" parameters

% define input vector
u  = [ Ax;   Az;   q    ];
u0 = [ rand; rand; rand ]; % "known" inputs

nstates = length(xaug); % length of augmented state vector

% define state transition function
f = [(Ax-Lx) - g*sin(th) - (q-Lq)*Vz;
     (Az-Lz) + g*cos(th) + (q-Lq)*Vx;
     (q-Lq);
     -Vx*sin(th)+Vz*cos(th);
     0;
     0;
     0 ];
 
% define state observation function
h = [sqrt(Vx^2+Vz^2); 
     th; 
     -ze];


% Calculate the first Lie derivative
%   calculate Jacobian of state observation function
Hx = simplify(jacobian(h, xaug));
%   construct complete Lie derivative
Hxf = Hx * f;

% Calculate the second Lie derivative
%   calculate Jacobian of Hx * f
Hxxf = simplify(jacobian(Hxf, xaug));
%   construct complete Lie derivative
Hxxff = Hxxf * f;

% Calculate the third Jacobian of the second Lie derivative
%   calculate Jacobian of Hx * f
Hxxxff = simplify(jacobian(Hxxff, xaug));

% Construct observability matrix:
Ox = [Hx; Hxxf; Hxxxff];

% calculate rank by substituting initial state values and known
% parameters/inputs
Obsnum  = (subs(Ox,     xaug, xaug0));
Obsnum  = (subs(Obsnum, u,    u0));
Obsnum  = (subs(Obsnum, p,    p0));

rankObs = double(rank(Obsnum));
fprintf('\t-> Rank of Observability matrix is %d\n', rankObs);
if (rankObs >= nstates)
    fprintf('Observability matrix is of Full Rank: the state is Observable!\n');
else
    fprintf('Observability matrix is NOT of Full Rank: the state is NOT Observable!\n');
end



