clear ; close all; clc
load('ex4data1.mat');
load('ex4weights.mat');
K = 10;
m = size(X, 1);
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
nn_params = [Theta1(:) ; Theta2(:)];
                 
    
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));   
                 
a1 = [ones(m, 1) X];
a2 = sigmoid(a1*Theta1');
a2 = [ones(size(a2, 1), 1) a2];
a3 = sigmoid(a2*Theta2');
h = a3;

y_m = zeros(m, K);
for i = 1:m 
  for j = 1:K 
    y_m(i, y(i)) = 1;
  end
end
  
J = (1/m)*((-y_m).*log(h)-(1.-y_m).*log(1.-h)); 
J = sum(J); 
J = sum(J);

%===================================================================


delta_3 = a3 - y_m;
teste = delta_3*Theta2;
delta_2 = teste(:, 2:end).*sigmoidGradient(a1*Theta1');

Theta_grad_2 = (1/m)*(a2'*delta_3)';
Theta_grad_1 = (1/m)*(a1'*delta_2)';

lambda = 0;
reg1_grad = (lambda/m)*Theta1;
reg2_grad = (lambda/m)*Theta2;
reg1_grad(1) = 0;
reg2_grad(1) = 0;


size(Theta_grad_1)
size(reg1_grad)
size(Theta_grad_2)
size(reg2_grad)

