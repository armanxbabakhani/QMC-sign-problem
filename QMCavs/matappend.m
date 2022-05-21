function mat = matappend(A , B) 
% mat = zeros(size(A,1), size(B,2));
% mat = num2cell(mat);
mat = cell(size(A,1), size(B,2));
if strcmp(class(A),'double')
    indices = find(A);
    C = cell(size(A));
    for i=1:length(indices)
        idx = indices(i);
        C{idx} = A(idx);
    end
    A = C;
end
for i=1:size(A,1)
    inda = find(~cellfun(@isempty,{A{i,:}}));
    E1 = A{i, inda};
    indb = find(B(inda,:));
    E2 = B(inda, indb);
    mat{i,indb} = [E1 E2];
end
end
        
    