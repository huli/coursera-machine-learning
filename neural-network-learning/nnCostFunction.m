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

% Calculating total cost
J = cost + regularization_cost;

% Calculating the gradient with backpropagation
delta_1 = 0;
delta_2 = 0;

for i = 1:m
  
    a1 = [1;  X(i, :)']; 
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)]; 
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Compute delta for output layer
    d3 = a3 - Y(i, :)';
    
    % Compute delta for hidden layer
    d2 = (Theta2(:, 2:end)' * d3) .* sigmoidGradient(z2);

    % Accumulate gradients
    delta_2 += (d3 * a2');
    delta_1 += (d2 * a1');
    
endfor

% Calculating gradient
Theta1_grad = (1 / m) * delta_1;
Theta2_grad = (1 / m) * delta_2;

% Adding regularization
Theta1_grad(:, 2:end) += ((lambda / m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2(:, 2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
