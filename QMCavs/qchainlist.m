function [DPdiags, ddiffs] = qchainlist(q, Ps , DPs , H0, invT)
%clist = [];
DPdiags = [];  % DPdiags returns a numberofcombintations by dim(H) array of 
               % diagonal elements of multiplication of D_iP_i

ddiffs = [];   % Rerturns a numberofcombintations by dim(H) array of the corresponding
               % divided difference of the chain of energies.
dim = length(Ps{1});

% Preparing H0*P_i = Diag(H)*P_i to extract the chain of energies.
for j = 1:length(Ps)
    H0Ps{j} = H0*Ps{j};
end

% Starting from initial chain of all P_1s
initial_chain = ones(1,q);
len = length(Ps);

for i = 0:len^q - 1
    b = str2double(regexp(dec2base(i, len), '\d', 'match')); % Using this step to generate vector of length
                                            % q to run through all the
                                            % possible combinations of P_i
                                            % of length q
    deltaC = [zeros(1, q-length(b)) b];
    curr_chain = initial_chain + deltaC;
    
    X = Ps{curr_chain(1)};
    for j = 2:q
        X = X*Ps{curr_chain(j)};
    end
    % Checking whether the multiplication of the chain generated is the
    % identity. If so, we extract the divided differences and the diagonal
    % entries of the corresponding D_iP_i chain.
    if all(abs(X - eye(dim)) < 1E-7) 
        %clist = [clist ; curr_chain];
        dps = DPs{curr_chain(1)};
        % Using a function called matappend to extract a chain of energies.
        E_chain = H0Ps{curr_chain(1)};
        E_chain = matappend(H0, E_chain);
        for k =2:length(curr_chain)
            dps = dps*DPs{curr_chain(k)};
            E_chain = matappend(E_chain, H0Ps{curr_chain(k)});
        end
        DPdiags = [DPdiags ; diag(dps)'];
        for k = 1:length(E_chain)
            E_chain{k,k} = divdiff(E_chain{k,k}, invT);
        end
        ddiffs = [ddiffs; cell2mat(E_chain)];
    end
end
end
