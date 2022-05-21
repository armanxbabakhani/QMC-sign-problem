function U = unitary(ns)
n = sqrt(sum(ns.^2,2));
n1 = n(1);
a = (ns(1,2) + 1i*ns(1,1))/n1;
b = 1i*ns(1,3)/n1;
U = [cos(n1)+ b*sin(n1), a*sin(n1); -conj(a)*sin(n1), cos(n1)- b*sin(n1)];
for k=2:size(ns,1)
    U = kron(U, [cos(n(k))+1i*ns(k,1)*sin(n(k))/n(k) (ns(k,2) + 1i*ns(k,1))*sin(n(k))/n(k); -(ns(k,2) - 1i*ns(k,1))*sin(n(k))/n(k), cos(n(k))-1i*ns(k,3)*sin(n(k))/n(k)]);
end
end