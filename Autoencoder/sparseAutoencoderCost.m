function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

m = double(size(data, 2));
% Forward propagation
Z2 = bsxfun(@plus, W1*data, b1);
A2 = sigmoid(Z2);
Z3 = bsxfun(@plus, W2*A2, b2);
A3 = sigmoid(Z3);

diff = A3 - data;
p = sparsityParam;
pHat = 1 / m * sum(A2, 2);

squareError = 0.0;
for i = 1:m
    squareError = squareError + diff(:, i)' * diff(:, i);
end

% cost = 0.5 / m * sum(diag(diff' * diff)) + lambda / 2 * (sum(sum((W1 .^ 2))) ...
%     + sum(sum(W2 .^ 2)));

cost = 0.5 / m * squareError + lambda / 2 * (sum(sum((W1 .^ 2))) + sum(sum(W2 .^ 2))) ...
    + beta * sum(p * log(p./pHat) + (1 - p) * log((1 - p)./(1-pHat)));

% Square error term
D3 = -1 * (data - A3) .* sigmoidGradient(Z3);
KL = beta * (-p./pHat + (1-p)./(1-pHat));
D2 = bsxfun(@plus, W2' * D3, KL) .* sigmoidGradient(Z2);
W2g = D3 * A2';
W1g = D2 * data';

W1grad = 1 / m * W1g + lambda * W1;
W2grad = 1 / m * W2g + lambda * W2;
b1grad = 1 / m * (D2 * ones(m , 1));
b2grad = 1 / m * (D3 * ones(m , 1));

%

%-------------------------------- -----------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
