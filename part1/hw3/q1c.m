clc
clear
rng(520) % reproduce seed
train_x=-1:0.05:1;
train_y=1.2*sin(pi*train_x)-cos(2.4*pi*train_x)+0.3*randn(1,41);
test_x=-1:0.01:1;
test_y=1.2*sin(pi*test_x)-cos(2.4*pi*test_x);
%% train
MSE_train = [];
MSE_test = [];
r_train = abs(train_x' - train_x);
RBF_train = exp(-r_train.^2./0.02);
r_test = abs(test_x' - train_x);
RBF_test = exp(-r_test.^2./0.02);
%% different hyparameter
lambda = 0;
w =pinv(RBF_train'*RBF_train+lambda*eye(size(RBF_train,2)))*RBF_train'*train_y';
pred_y_train = (RBF_train*w)';
MSE_train = [MSE_train; mse(pred_y_train,train_y)];
pred_y_test_0 = (RBF_test*w)';
MSE_test = [MSE_test;mse(pred_y_test_0,test_y)];
%% 
lambda = 0.01;
w =pinv(RBF_train'*RBF_train+lambda*eye(size(RBF_train,2)))*RBF_train'*train_y';
pred_y_train = (RBF_train*w)';
MSE_train = [MSE_train; mse(pred_y_train,train_y)];
pred_y_test_001 = (RBF_test*w)';
MSE_test = [MSE_test;mse(pred_y_test_001,test_y)];
%% 
lambda = 0.1;
w =pinv(RBF_train'*RBF_train+lambda*eye(size(RBF_train,2)))*RBF_train'*train_y';
pred_y_train = (RBF_train*w)';
MSE_train = [MSE_train; mse(pred_y_train,train_y)];
pred_y_test_01 = (RBF_test*w)';
MSE_test = [MSE_test;mse(pred_y_test_01,test_y)];
%% 
lambda = 1;
w =pinv(RBF_train'*RBF_train+lambda*eye(size(RBF_train,2)))*RBF_train'*train_y';
pred_y_train = (RBF_train*w)';
MSE_train = [MSE_train; mse(pred_y_train,train_y)];
pred_y_test_1 = (RBF_test*w)';
MSE_test = [MSE_test;mse(pred_y_test_1,test_y)];
%% 
lambda = 10;
w =pinv(RBF_train'*RBF_train+lambda*eye(size(RBF_train,2)))*RBF_train'*train_y';
pred_y_train = (RBF_train*w)';
MSE_train = [MSE_train; mse(pred_y_train,train_y)];
pred_y_test_10 = (RBF_test*w)';
MSE_test = [MSE_test;mse(pred_y_test_10,test_y)];
%% 
lambda = 100;
w =pinv(RBF_train'*RBF_train+lambda*eye(size(RBF_train,2)))*RBF_train'*train_y';
pred_y_train = (RBF_train*w)';
MSE_train = [MSE_train; mse(pred_y_train,train_y)];
pred_y_test_100 = (RBF_test*w)';
MSE_test = [MSE_test;mse(pred_y_test_100,test_y)];
%%  Plot
fig = figure();
hold on
plot(train_x,train_y,'o')
plot(test_x,test_y)
plot(test_x,pred_y_test_0,'LineWidth',1.2)
plot(test_x,pred_y_test_001,'LineWidth',1.2)
plot(test_x,pred_y_test_01,'LineWidth',1.2)
plot(test_x,pred_y_test_1,'LineWidth',1.2)
plot(test_x,pred_y_test_10,'LineWidth',1.2)
plot(test_x,pred_y_test_100,'LineWidth',1.2)
legend('Training set (with noise)', 'Test set (without noise)','factor=0','factor=0.01','factor=0.1','factor=1','factor=10','factor=100')
title('Exact Interpolation with different regularization factor')
hold off