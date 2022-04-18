%This experiment is a concrete implementation of ERWNMF.
%	Usage:
%	[T, W, H, Func] = EWEucNMF(V, W, H, MaxIter,Gamma)
%   Input：
%         V - Image matrix.
%         W - Base matrix.
%         H - Coefficient matrix.
%         MaxIter - Number of iterations.
%         Gamma - Hyperparameters of entropy regularization.
%   Ouput:
%         T - Weight matrix.
%         W - The base matrix after iteration.
%         H - The Coefficient matrix after iteration.
%         Func - The value of the loss function.

% T.* ||V - W*H|| + Gamma T.*ln(T)

%%
function [T, W, H, Func] = EWEucNMF(V, W, H, MaxIter,Gamma)
% fprintf('EWEucNMF\n')
if size(V) ~= size(W*H)
    fprintf('incorrect size of W or H\n')
end
[RowNum, ColNum] = size(V);
[ReduceDim, ~] = size(H);

Func = zeros(MaxIter, 1);

for i = 1:MaxIter
    T = sum( (V - W * H).^2, 2 );
    T = exp( - T / Gamma );  
    T = T ./ sum(T);

    W = W .* ( V * H') ./ ( W * H * H' +eps);     
    TR = repmat(T, 1, ReduceDim);
    TW = TR .* W;
    H = H .* ( TW' * V ) ./ (TW' * (W * H) + eps);  
    Func(i) =  0.5*sum(T.*sum((V - W * H).^2, 2)) + Gamma * sum(sum(T.*log(T+eps)));
end
return;