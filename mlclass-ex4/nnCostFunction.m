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

Y = zeros(length(y),num_labels);
for i = 1:length(y) 
    Y(i,y(i))=1;
endfor;
y=Y;
w=X
X = [ ones(size(X,1),1) X];
z1 = X*Theta1'; 
a1 = sigmoid(z1);
a1 = [ ones(size(a1,1),1) a1];
z2 = a1 * Theta2';
a2 = sigmoid(z2);
diff_sum=0;


%TODO need to vectorize this.
for i = 1:m;
    aa = y(i,:);
    bb = a2(i,:);
    diff_sum += aa*log(bb)' +(1-aa) * log(1-bb)'; 
endfor;

diff_sum =-(diff_sum/m);
diff_sum = diff_sum + lambda*( sum(nn_params.^2)-sum(Theta1(:,1).^2) -sum(Theta2(:,1).^2)) /(2*m);
disp(diff_sum);
J = diff_sum;

d3 = a2 - y;

for i = 1:m
    z_2 = z1(i,:);
    d_3 =d3(i,:);
    a_2 = a1(i,:);
    a_1 = X(i,:);
    disp("THETA2 ,d3, z_2") , disp(size(Theta2)),disp(size(d_3)), disp(size(z_2))
    d_2 = (Theta2' *d_3') ; 
    d_2 = d_2(2:end);
    d_2 = d_2.*sigmoidGradient(z_2');
    t1=d_3' *a_2;
    Theta2_grad += t1;  
    t2= d_2 *a_1;
    Theta1_grad +=t2; 
endfor
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
t1 = Theta1;
t1(:,1)=zeros(size(t1,1),1);
Theta1_grad += (lambda/m)*t1;
t2=Theta2;
t2(:,1)=zeros(size(t2,1),1);

Theta2_grad +=(lambda/m)*t2;
    % ====================== YOUR CODE HERE ======================


% Instructions: You should complete the code by working through the
%               following parts.



% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
