function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
Pred_diff = (((X * Theta') .* R) - Y);
J = (Pred_diff .^ 2);
J = .5 * sum(J(:));
J = J + (lambda/2)*( sum(sum(X .^ 2)) + sum(sum(Theta .^ 2)));

X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

for i = 1:size(X,1)
    X_grad(i,:) = sum(Pred_diff(i,:)' .* Theta) + lambda*X(i,:);
end
for j = 1:size(Theta,1)
    Theta_grad(j,:) = sum(Pred_diff(:,j) .* X) + lambda*Theta(j,:);
end

grad = [X_grad(:); Theta_grad(:)];


end
