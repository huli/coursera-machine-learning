function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% First some forward propagation 
a_1_1 = [ones(size(X, 1), 1) X];
z_2 = a_1_1 * Theta1';
a_2 = sigmoid(z_2);
a_2_1 = [ones(size(X,1), 1) a_2];
z_3 = a_2_1 * Theta2';
a_3 = sigmoid(z_3);

% Computing cost
Y = eye(num_labels)(y, :);
cost = (1 / m) * sum(sum((-Y .* log(a_3) - (1- Y) .* log(1- a_3)))); 

% Computing regularization cost
regularization_cost = (lambda/ (2*m)) * (sum(sum(Theta1(:, 2:end) .**2)) ...
                              + sum(sum(Theta2(:, 2:end) .**2)));

J = cost + regularization_cost;



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
