function ratio = avsgnnew(H, invT , convidx)
% Extracting the permutation and diagonal entries:
[Ps, DPs] = Pextractor(H);
H0 = diag(diag(H));

% Setting the cutoff qmax:
qmax = int32(invT) + convidx;

% q != 0 walks. Note that there is no q=1 walk.
y = 0;
yabs = 0;
for q = 2:qmax
    [dpdiags , ddiffs] = qchainlist(q, Ps , DPs , H0, invT);
    multed = real(dpdiags.*ddiffs);
    y = y + sum(sum(multed));
    yabs = yabs + sum(sum(abs(multed)));
end

% q = 0 walk:
for i= 1:length(H0)
    y = y + real(divdiff(H0(i,i), invT));
    yabs = yabs + abs(real(divdiff(H0(i,i), invT)));
end
ratio = y/yabs;
end
