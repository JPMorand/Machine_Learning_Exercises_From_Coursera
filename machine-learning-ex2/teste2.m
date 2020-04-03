clear ; close all; clc
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

X = mapFeature(X(:,1), X(:,2));

theta = zeros(size(X, 2), 1);
test_theta = ones(size(X,2),1);

lambda = 1;



m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X*theta);
h
grad(1) = (1/m)*(X'(1,:))*(h_theta - y);

grad(2:size(theta,1)) = 1/m * (X'(2:size(X',1),:)*(h_theta - y) + lambda*theta(2:size(theta,1),:));
grad
%grad(2:length(grad),:) = 1/m .* sum((h(2:length(h),:).-y(2:length(y),:)).*X(:, 2:end))';

%grad = 1/m * sum((h-y).*X )+(lambda/(m))*theta;
%grad(1,:) = 1/m *((h(1,:)-y(1,:))*X(1,:));
%[nr, nc] = size(h)