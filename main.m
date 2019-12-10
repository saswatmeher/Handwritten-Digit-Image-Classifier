clear;
% set the structure of the neural network and set output as vector;

input_layer_size = 400;
hidden_layer_size = 150;
num_labels = 10;
lambda = 1;
P = 0.70 ;

load('dataset.mat');

%load the data and separate it as 80 % as training data and 20 % as cross validation data.

%----------------------------------------------------------------------------

[m,n] = size(X) ;

I = eye(num_labels);
Y = zeros(m,num_labels);
for i = 1 : m
  Y(i,:) = I(y(i),:);
end


idx = randperm(m)  ;
X_train = X(idx(1:round(P*m)),:) ; 
Y_train = Y(idx(1:round(P*m)),:) ;
X_val = X(idx(round(P*m)+1:end),:) ;
Y_val = Y(idx(round(P*m)+1:end),:) ;

%---------------------------------------------------------------------------

% training the neural network;

[theta1 theta2] = trainNN(X_train,Y_train,input_layer_size,hidden_layer_size,num_labels,lambda);

% get the error for the trained data 

err = calcError(X_val,Y_val,theta1,theta2)
disp(err)
%-------------------------------------------------------------------

% calculate the cross validation error and use learningCurve to plot carious curves;


learningCurve(X_train, Y_train, X_val, Y_val,input_layer_size,hidden_layer_size,num_labels, lambda);

degreeCurve(X_train, Y_train, X_val, Y_val,input_layer_size,num_labels, lambda);

validationCurve(X_train, Y_train, X_val, Y_val,input_layer_size,hidden_layer_size,num_labels);

clear;



