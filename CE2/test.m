%---------------------------------------------------------------
% SF2527 Computer Exercise 2, Part 1(b)
% Comparison of MATLAB ODE solvers for 1D heat equation
%---------------------------------------------------------------

clear; close all; clc;

d = 0.35; a = 1.2; L = 1; tau_end = 2;
bc_left = @(t) (t <= a).*sin(pi*t/a);
N_values = [100, 200, 400];

Results = zeros(length(N_values), 7); % <-- 7 columns now

opts_base = odeset('RelTol',1e-4,'AbsTol',1e-6);

for i = 1:length(N_values)
    N = N_values(i);
    dx = L / N;
    x = linspace(0, L, N+1)';

    % Construct A matrix for du/dt = A*u + b(t)
    e = ones(N+1,1);
    A = spdiags([e -2*e e], -1:1, N+1, N+1);
    A(1,:) = 0; A(1,1) = 1;          % Dirichlet at x=0
    A(end,end-1) = 2; A(end,end) = -2; % Neumann at x=1
    A = d/dx^2 * A;

    % ODE system: du/dt = A*u + b(t)
    rhs = @(t,u) A*u + b_func(t,a,x,N);

    % Initial condition
    u0 = zeros(N+1,1);

    % --- ODE23 (explicit) ---
    tic;
    [T1,U1] = ode23(rhs, [0 tau_end], u0);
    time1 = toc;
    nsteps1 = length(T1);

    % --- ODE23s (implicit) ---
    tic;
    [T2,U2] = ode23s(rhs, [0 tau_end], u0);
    time2 = toc;
    nsteps2 = length(T2);

    % --- ODE23sJ (implicit with Jacobian) ---
    opts = odeset('Jacobian',A);
    tic;
    [T3,U3] = ode23s(rhs, [0 tau_end], u0, opts);
    time3 = toc;
    nsteps3 = length(T3);

    % Store results
    Results(i,:) = [N nsteps1 nsteps2 nsteps3 time1 time2 time3];

    % Display progress
    fprintf('N=%d done.\n', N);
end

% Display summary table
fprintf('\nResults:\n');
fprintf('N     nsteps(ode23)  nsteps(ode23s)  nsteps(ode23sJ)   time(ode23)   time(ode23s)   time(ode23sJ)\n');
disp(Results);

% Helper function for source term b(t)
function b = b_func(t,a,x,N)
    b = zeros(N+1,1);
    if t <= a
        b(1) = sin(pi*t/a);
    else
        b(1) = 0;
    end
end
