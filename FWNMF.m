%This experiment is a concrete implementation of FWWNMF
%	Usage:
%	[T, W, H, Func] = FWEucNMF(V, W, H, MaxIter,Power)
%   Input：
%         V - Image matrix.
%         W - Base matrix.
%         H - Coefficient matrix.
%         MaxIter - Number of iterations.
%         Power - Index of the weight matrix.
%   Ouput:
%         T - Weight matrix.
%         W - The base matrix after iteration.
%         H - The Coefficient matrix after iteration.
%         Func - The value of the loss function.
          
% T.* ||V - W*H|| 
%%
function [T, W, H, Func] = FWEucNMF(V, W, H, MaxIter,Power)
% fprintf('FWEucNMF\n')
if size(V) ~= size(W*H)
    fprintf('incorrect size of W or H\n')
end
[RowNum, ColNum] = size(V);
[ReduceDim, ~] = size(H);

Func = zeros(MaxIter, 1);

for i = 1:MaxIter          
    T = sum( (V - W * H).^2, 2 );
    T = 1 ./ (nthroot(T, Power - 1) + eps);
    T = T ./ sum(T);
    
    W = W .* ( V * H') ./ ( W * H * H' +eps);     
    Tm = T.^Power;
    TR = repmat(Tm, 1, ReduceDim);
    TW = TR .* W;
    H = H .* ( TW' * V ) ./ (TW' * W * H + eps);
%     PPP=sum((V - W * H).^2);
    
    Func(i) =  sum(Tm .* sum((V - W * H).^2, 2));
end

return;

