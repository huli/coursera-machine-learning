function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


errors = zeros(64, 3);
values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

counter = 1;
for c = 1:8
    for s = 1:8
        
        % train model with current parameters on training set
        model= svmTrain(X, y, values(c), @(x1, x2) gaussianKernel(x1, x2, values(s)));
        
        % predict classes on cross validation set
        predictions = svmPredict(model, Xval);
        
        % calculate current error and store with parameters
        errors(counter, :) = [values(c) values(s) mean(double(predictions ~= yval))];
        
        counter += 1;
    endfor
endfor

% get parameters of smalles errors
[val, index] = min(errors(:, 3));

C = errors(index, 1);
sigma = errors(index, 2);

end
