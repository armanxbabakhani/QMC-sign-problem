function Ru = randunitary(n)

X = (randn(n) + 1i*randn(n))/sqrt(2);
[Q,R] = qr(X);
R = diag(diag(R)./abs(diag(R)));
Ru = Q*R;

end