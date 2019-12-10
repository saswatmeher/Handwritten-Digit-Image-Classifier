function [Theta1, Theta2] = trainNN(X,y,input_layer_size, ...
                                    hidden_layer_size,num_labels,lambda);


options = optimset('MaxIter', 20);

% create a neural network parameter vector containing all the random weights 
Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size,num_labels);
initial_nn_params = [Theta1(:);Theta2(:)];


% Create "short hand" for the cost function to be minimized

costFunction = @(p) NNCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
end