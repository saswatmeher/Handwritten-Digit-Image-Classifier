function g = sigmoidGradient(z)

%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z

g = zeros(size(z));

g = (1.0 ./ (1.0 + exp(-z))) .* (1-(1.0 ./ (1.0 + exp(-z))));

end
