function Hnew = u_trans(H, U)

Hnew = U*H*ctranspose(U);
end