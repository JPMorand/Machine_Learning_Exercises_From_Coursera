function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
steps = [0.01,0.03,0.1,0.3,1,3,10,30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
k = 1;
for i = 1:length(steps)
  for j = 1:length(steps)
    model= svmTrain(X, y, steps(i), @(x1, x2) gaussianKernel(x1, x2, steps(j))); 
    predictions = svmPredict(model, Xval);
    error(k, :) = [mean(double(predictions ~=yval)), steps(i), steps(j)]; 
    k = k+1;
  end
end  

[M, I] = min(error(:, 1), [], 1);
C = error(I, 2);
sigma = error(I, 3);




% =========================================================================

end
