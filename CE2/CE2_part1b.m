%---------------------------------------------------------------
% SF2527 Computer Exercise 2, Part 1(b)
% Comparison of MATLAB ODE solvers for 1D heat equation
% (Interior nodes only, BCs handled explicitly)
%---------------------------------------------------------------

clear; close all; clc;

d = 0.35; a = 1.2; L = 1; tau_end = 2; % Problem set up
alpha = @(t) (t <= a).*sin(pi*t/a);   % left Dirichlet BC
N_values = [100, 200, 400];

% Results = zeros(length(N_values), 7);
Results = struct();

% Tolerances (can adjust if too strict)
opts_base = odeset('RelTol',1e-3,'AbsTol',1e-6);

for i = 1:length(N_values)
    N = N_values(i);
    dx = L / N;
    x = linspace(0, L, N+1)';

    % Construct matrix for interior nodes only (2..N)
    e = ones(N-1,1);
    A = spdiags([e -2*e e], -1:1, N-1, N-1);

    % Modify last row for Neumann BC at right boundary (u_N+1 = u_N)
    A(end,end-1) = 2;
    A(end,end)   = -2;

    A = d/dx^2 * A;

    % RHS for interior system
    rhs = @(t,u_interior) interior_rhs(t,u_interior,A,alpha,d,dx);

    % Initial condition (zeros for interior nodes)
    u0 = zeros(N-1,1);

    % --- ODE23 ---
    tic;
    [T1,U1] = ode23(rhs, [0 tau_end], u0, opts_base);
    time_ode23 = toc;  nsteps_ode23 = length(T1);

    % --- ODE23s ---
    tic;
    [T2,U2] = ode23s(rhs, [0 tau_end], u0, opts_base);
    time_ode23s = toc;  nsteps_ode23s = length(T2);

    % --- ODE23s with Jacobian ---
    optsJ = odeset(opts_base,'Jacobian',A);
    tic;
    [T3,U3] = ode23s(rhs, [0 tau_end], u0, optsJ);
    time_ode23sJ = toc;  nsteps_ode23sJ = length(T3);

    % Store results
    Results(i).N = N;
    Results(i).ode23.nsteps = nsteps_ode23;
    Results(i).ode23.time = time_ode23;
    Results(i).ode23s.nsteps = nsteps_ode23s;
    Results(i).ode23s.time = time_ode23s;
    Results(i).ode23sJ.nsteps = nsteps_ode23sJ;
    Results(i).ode23sJ.time = time_ode23sJ;
    
end

Nvals = [Results.N]';

nsteps_ode23   = arrayfun(@(r) r.ode23.nsteps, Results)';
nsteps_ode23s  = arrayfun(@(r) r.ode23s.nsteps, Results)';
nsteps_ode23sJ = arrayfun(@(r) r.ode23sJ.nsteps, Results)';

time_ode23   = arrayfun(@(r) r.ode23.time, Results)';
time_ode23s  = arrayfun(@(r) r.ode23s.time, Results)';
time_ode23sJ = arrayfun(@(r) r.ode23sJ.time, Results)';

T = table(Nvals, ...
          nsteps_ode23, nsteps_ode23s, nsteps_ode23sJ, ...
          time_ode23, time_ode23s, time_ode23sJ);

T.Properties.VariableNames = { ...
    'N', ...
    'Steps_ode23', 'Steps_ode23s', 'Steps_ode23sJ', ...
    'Time_ode23', 'Time_ode23s', 'Time_ode23sJ'};

disp(T);

% ---------- Helper function ----------
function dudt = interior_rhs(t,u_interior,A,alpha,d,dx)
    % Compute RHS for interior unknowns
    dudt = A*u_interior;

    % Add contribution of left boundary (Dirichlet) to first interior node
    u_left = alpha(t);
    dudt(1) = dudt(1) + (d/dx^2) * u_left;
end
