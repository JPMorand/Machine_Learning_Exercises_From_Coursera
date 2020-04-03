clear ; close all; clc
load ('ex5data1.mat');
theta = [1; 1];

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
lambda = 1;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

X = [ones(m, 1) X]
h = X*theta;
J = 1/(2*m)*sum(h-y);
theta_reg = theta(2:end);
reg = lambda/(2*m)*sum(theta_reg.^2);
J = J + reg;
