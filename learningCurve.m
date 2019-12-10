function [] = learningCurve(X, y, Xval, yval,input_layer_size,hidden_layer_size,num_label, lambda)

   
%LEARNINGCURVE Generates the train and cross validation set errors needed 
% to plot a learning curve and it returns the train and
% plot those cross validation set errors and train errrs


% Number of training examples
m = size(X, 1);

% error values
error_train = zeros(m/100, 1);
error_val   = zeros(m/100, 1);


m_val = length(yval);
for i = 100:100:m
  x_temp = [X(1:i,:)];
  y_temp = y(1:i,:);
  [theta1 theta2] = trainNN(x_temp, y_temp, input_layer_size, hidden_layer_size,num_label, lambda);
  error_train(i) = calcError(x_temp,y_temp,theta1,theta2);
  error_val(i) = calcError(Xval, yval, theta1,theta2);
end


plot(1:size(error_train,1), error_train, 1:size(error_val,1), error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

fprintf('program is paused, press enter to continue;\n');
pause;

end
