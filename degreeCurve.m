% This is the function to test out model with degree curve that have size of hidden layer
% and error as x and y axis for each train and test data

function [] = degreeCurve (X, y, Xval, Yval,input_layer_size,num_label, lambda)
   hidden_layer_vec = [10; 30; 50;70;100;150;200;300];
   n = size(hidden_layer_vec,1);
   
   error_train = zeros(n,1);
   error_val = zeros(n,1);
   m = size(X,1);
   m_val = size(Xval,1);
   for i = 1 : n
      [theta1 theta2] = trainNN(X, y, input_layer_size, hidden_layer_vec(i),num_label, lambda);
      error_train(i) = calcError(X,y,theta1,theta2);
      error_val(i) = calcError(Xval, Yval, theta1,theta2);
   end
   
   plot(hidden_layer_vec, error_train,hidden_layer_vec, error_val);
   title('Degree curve for linear regression with varing degree')
   legend('Train', 'Cross Validation')
   xlabel('size of hidden layer')
   ylabel('Error')
end
