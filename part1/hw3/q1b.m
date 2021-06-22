clc
clear
rng(520) % reproduce seed
train_x=-1:0.05:1;
train_y=1.2*sin(pi*train_x)-cos(2.4*pi*train_x)+0.3*randn(1,41);
test_x=-1:0.01:1;
test_y=1.2*sin(pi*test_x)-cos(2.4*pi*test_x);
%% train
rand_cens = datasample(train_x,20,2);
r = abs(train_x' - rand_cens);
dmax=max(max(dist(rand_cens',rand_cens)));
RBF = exp(-20*r.^2./dmax.^2);
w = pinv(RBF)*train_y';
% Train accuracy
pred_y_train = (RBF*w)';
MSE_train = mse(pred_y_train,train_y);
%% test
r = abs(test_x' - rand_cens);
RBF = exp(-20*r.^2./dmax.^2);
% Test accuracy
pred_y_test = (RBF*w)';
MSE_test = mse(pred_y_test,test_y);
%% Plot test set simulation
figure
hold on 
plot(train_x,train_y,'o')
plot(test_x,test_y)
xlabel('x'); ylabel('y'); title('Fixed centres selected at random'); 
plot(test_x,pred_y_test,'Linewidth',1.4);
grid;
legend('Training set (with noise)', 'Test set (without noise)','Approximated function');