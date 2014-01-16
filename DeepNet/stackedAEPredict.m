function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
% forward feed
R = zeros(hiddenSize, size(data, 2));
for d = 1:numel(stack)
    cw = stack{d}.w;
    cb = stack{d}.b;
    if d == 1
        R = bsxfun(@plus, cw*data, cb);
    else
        R = bsxfun(@plus, cw*R, cb);
    end
    R = sigmoid(R);
end

% SI: input to softmax classifier
P = softmaxTheta * R;
P = exp(bsxfun(@minus, P, max(P, [], 1)));
P = bsxfun(@rdivide, P, sum(P));

[y, pred] = max(P);
% -----------------------------------------------------------

end

