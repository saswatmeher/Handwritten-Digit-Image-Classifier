function W = randInitializeWeights(L_in, L_out)

%   RANDINITIALIZEWEIGHTS randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms



% initialisation

W = zeros(L_out, 1 + L_in);

% random initialisation using epsilon as a parameter

epsilon_init = 0.12;
W = (rand(L_out, 1 + L_in) * 2 * epsilon_init) - epsilon_init;






% =========================================================================

end
