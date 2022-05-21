%% Summary of the results
% Using simulated annealing we can search and find the parameters 'ns' with
% the lowest cost.

% The cost function is set to the negative of the average of the
% Hamiltonian with qmax = 3;

% Simulated annealing with the given cost function allows one to find
% unitaries that drastically improve the average sign of the initial
% Hamiltonian.
%% qubit operators
X = [0 1; 1 0];
Z = [1 0; 0 -1];
Y = Z*X;
I = eye(2);

%% 3-qubit Hamiltonian
H3 = kron(X,kron(X,I)) + kron(X,kron(I,X)) + kron(I,kron(X,X)) - (kron(Z,kron(Z,I)) + kron(Z,kron(I,Z)) + kron(I,kron(Z,Z)));
Ht = H3;
[Ps , DPs] = Pextractor(H3);
invT = 1;
%% The average sign 
av_sign_initial = avsgnnew(Ps, DPs, diag(diag(H3)) , invT , 7);

%% Parameters for gradient descent!
beta = 50;
a = 1E-5;

pars = [beta, a];
nosteps = 500;
showcost = 1;
%% Simulated Annealing
% Compare 'av_sign_final_0' with 'av_sign_initial' to see if minimizing
% using simulated annealing has improved the average sign.
ns = simanneal(Ht, invT, 500,0,1);

%% Basis change using the 'ns' found!

UHt = u_trans(Ht, unitary(ns(4:6,:)));

%% Average sign after simulated annealing

avsign_sa = avsgnnew(UHt , invT , 5);

%% Gradient Descent
% Compare 'av_sign_final_1' with 'av_sign_initial' and 'av_sign_final_0' to see if minimizing
% using gradient descent has improved the average sign.
ns_best = ns;
[U, ns_sub, finalcost] = runopt(Ht, ns(1:3,:), pars, nosteps, showcost);
av_sign_final_1 = avsgn(u_trans(Ht, U), invT);