function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

h =  X * theta;
J = (1/(2 * m)) * (sum((h - y) .** 2)) ...
        + (lambda/(2 * m)) * sum((theta(2:end) .**2));

j_0 = (1/m) * (sum((h -y)' * X(:, 1)));
j_n = (1/m) * ((h -y)' * X(:, 2:end))' ...
          +  (lambda/m) .* theta(2:end) ;
     
grad = [j_0; j_n];
  

end
