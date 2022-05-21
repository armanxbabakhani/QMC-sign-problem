function y = cost_function_new(H, invT)
% This cost function only takes in the average sign of the 3 cycles!
[Ps, DPs] = Pextractor(H);
y = -avsgnnew(Ps, DPs, diag(diag(H)) , invT , 2);

% y = -avsgn(H, invT, 3);

%[sn, E_list] = ncyclefull(H,3); 
% dds = zeros(size(E_list,1),1);
% for i = 1:size(E_list,1)
%     dds(i) = divdiff(E_list(i,:),invT);
% end
% snreals = real(sn);
% x = [];
% if any(snreals)
%     x = [x; real(sn).*dds];
%     y = sum(x)/sum(abs(x));
% else
%     y = 0;
% end
% y = -y; % To force positive average signs!
end