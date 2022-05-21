function y = divdiff( Echain , invT )
%Start of the new code:
if length(Echain) ~= size(Echain,2)
    Echain = Echain' ;
end
dim = length(Echain);
mu = mean(-invT*Echain);
zz = -invT*Echain - mu;
n = dim - 1;
NN = n + 30; % Where is 30 coming from? Is it just a maximum dimension that we know won't be exceeded?
Fm = repmat(zz, n + 1 , 1);
F = Fm - Fm';
s = max( ceil(max(max(abs(F)))/3.5) , 1); % Question: Where does 3.5 come from?!
d = []; curr = 1;
for i=1:NN+1
    d = [d, curr];
    curr = curr/s;
end

for j = n:-1:0
    for k = NN:-1:n-j+1
        d(k) = d(k) + zz(j+1)*d(k+1)/k;
    end
    for k = n-j:-1:1
        d(k) = d(k) + F(k+j+1,j+1)*d(k+1)/k;
    end
    for m =1:n-j+1
        F(j+1,j+m) = d(m)*nchoosek(j+m-1,j);
    end
end
F = triu(F);
d = exp(mu)*F(1,:);
for k = 1:s-1
    d = d*F;
end
q = dim-1;
% if invT
%     y = d(end)*(-invT)^q/factorial(q);
% else
%     y = (-1)^(q)/factorial(q);
% end
y = d(end)*(-invT)^q/factorial(q);
end