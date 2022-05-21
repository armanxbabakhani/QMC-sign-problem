function nsfin = simanneal(H, invT, smax, graddesc, showcost)
dim = int16(log2(length(H)));

ns = 4*pi*rand(dim,3) - 2*pi;
U = unitary(ns);
currentcost = cost_function_new(u_trans(H,U),invT);

% ..... Keeping the best #top_no# ns!
top_no = 5; 
ns_size = size(ns,1);
nsfin = zeros(top_no*ns_size,3);
% ......

% Only in presence of gradient descent:
beta = 10;
a = 0.01;
pars = [beta, a];

best_cost = 1000;

for i=0:smax
    % Setting the temperature
    T = 0.5*(1 - i/smax);
    
    % Generating the next random unitary
    ns = 20*rand(dim,3) - 10;
    Ut = unitary(ns);
    Ht = u_trans(H,Ut);
    
    % Calculating the difference between the new and the current cost
    deltaC = cost_function_new(Ht, invT) - currentcost;
    
    if rand > prob(deltaC, T) || deltaC < 0
        % Only recording the ns of the all-time low costs
        if best_cost > currentcost
            best_cost = currentcost;
            nsfin = [ns ; nsfin(1:(top_no-1)*ns_size,:)];
        end
        U = Ut;
        
        % For purposes of printing the cost during runtime
        if showcost
            currentcost = cost_function_new(u_trans(H,U),invT)
        else
            currentcost = cost_function_new(u_trans(H,U),invT);
        end
    elseif graddesc
        % Only for running gradient descent during simulated annealing!
        [U, nsfin, currentcost] = runopt(H, nsfin, pars, 300 - 590*T, 1);
    end
    if ~any(real(ncycle(u_trans(H,U),3)))
        fprintf('found it!')
        break
    end
end
end
        
