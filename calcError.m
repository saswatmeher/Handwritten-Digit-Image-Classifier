function [x] = calcError(X,y,Theta1,Theta2)

m = size(X,1);
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2*Theta2';
hx = sigmoid(z3);

idx = zeros(m,1);

[dummy idx] = max(hx,[],2);

x = zeros(1,1);
for i = 1 : m
 x = x + (y(i,idx(i)) == 0);
end

%x = x/m;
end
