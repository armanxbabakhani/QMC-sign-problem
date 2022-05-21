function [Ps , DPs] = Pextractor(H)
[x , y] = find(H-diag(diag(H)));
[~, indices] = unique(y);
num = int32(length(y)/length(indices));
for i=1:num
    ps = zeros(length(H));
    dps = ps;
    ycurr = y;
    xcurr = x;
    for j = 1:length(H)
        inds = find(ycurr == j);
        yind = inds(1);
        xnow = xcurr(yind);
        ynow = ycurr(yind);
        xxnow = find(x == xnow);
        yynow = find(y == ynow);
        knidx = xxnow(find(xxnow == yynow));
        x = [x(1:knidx-1); x(knidx+1:end)];
        y = [y(1:knidx-1); y(knidx+1:end)];
        ps(xnow, ynow) = 1;
        dps(xnow, ynow) = H(xnow, ynow);
        
        restind = find(xcurr ~= xnow);
        ycurr = ycurr(restind);
        xcurr = xcurr(restind);
    end
    Ps{i} = ps;
    DPs{i} = dps;
end
        
        