function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% for i=1:m
%     Hx = 1/(1+exp(theta' * X(i,:)'));
%     Hx
%     Y = y(i)
%     log10(Hx)
%     log(1-Hx)
%     training_const(i)= -y(i) * log10(Hx) - (1-y(i)) * log10(1-Hx) 
%     pause
% endfor

calc = sigmoid(theta' * X')';
Cost = -(y' * log(calc) + (1-y') * log(1-calc));
J = Cost / m;
grad = X' * (calc - y) / m

% =============================================================

end
