function [] = ...
    validationCurve(X, y, Xval, Yval, input_layer_size, hidden_layer_size,num_label)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda the regularization factor


% Selected values of lambda (you should not change this)
lambda_vec = [0.01 0.03 0.1 0.2 0.3 1 3 ];

% error vectors
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

  m = size(X,1);
  m_val = size(Xval,1);
  for i = 1 : length(lambda_vec)
      [theta1 theta2] = trainNN(X, y, input_layer_size, hidden_layer_size,num_label, lambda_vec(i));
      error_train(i) = calcError(X,y,theta1,theta2);
      error_val(i) = calcError(Xval, Yval, theta1,theta2);
  end
   
   
   % plot --------------
   plot(lambda_vec, error_train,lambda_vec, error_val);
title('Validation curve for linear regression with varing degree')
legend('Train', 'Cross Validation')
xlabel('value of lambda')
ylabel('Error')






% =========================================================================

end
