clear 
clc
sigma = 0.1;
train_x=-1:0.05:1;
train_y=1.2*sin(pi*train_x)-cos(2.4*pi*train_x)+0.3*randn(1,41);
test_x=-1:0.01:1;
test_y=1.2*sin(pi*test_x)-cos(2.4*pi*test_x);
%% Calculate interpolation matrix and weights
r = abs(train_x' - train_x);
RBF = exp(-r.^2./0.02);
w = pinv(RBF)*train_y';
pred_y_train = (RBF*w)';
%% test
r = abs(test_x' - train_x);
RBF = exp(-r.^2./0.02);
pred_y_test = (RBF*w)';
%% Plot test set simulation
figure
hold on 
plot(train_x,train_y,'o')
plot(test_x,test_y)
xlabel('x'); ylabel('y'); title('Exact Interpolation'); 
plot(test_x,pred_y_test,'Linewidth',1.4);
grid;
legend('Training set (with noise)', 'Test set (without noise)','Approximated function');
%% Find sum of squared error (performace)
mse_test = mse(pred_y_test,test_y);
mse_train = mse(pred_y_train,train_y);

