% add plot data
clear;
clc;
train_x=-2:0.05:2;
train_y=1.2*sin(pi*train_x)-cos(2.4*pi*train_x);
test_x=-2:0.01:2;
test_y=1.2*sin(pi*test_x)-cos(2.4*pi*test_x);
all_x=[train_x test_x];
all_y=[train_y test_y];
%% net
net = feedforwardnet(10,'trainbr');
net = configure(net,all_x,all_y);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(train_x,2)
net.divideParam.testInd= size(train_x,2)+1:size(all_x,2);
net.trainParam.lr=0.001;
net.performFcn = 'mse';
net.trainParam.goal=0.001;
net.trainParam.epochs = 1000;

%% batch train
[net, tr] = train(net,all_x, all_y);
% view(net)
% plotregression(targets,outputs) plots the linear regression of targets relative to outputs.
%% result
pred_test = net(test_x);
pred_train = net(train_x);
perf_train = perform(net,pred_train,train_y);
perf_test = perform(net,pred_test,test_y);
figure(1);
xlabel('x');
ylabel('y');
plot(test_x,test_y,'Linewidth',3);
hold on
plot(test_x,pred_test,'Linewidth',1);
legend('desired result','test set output');
title('N=100')
hold off
y3=1.2*sin(pi*3)-cos(2.4*pi*3);
ne_y3=1.2*sin(pi*-3)-cos(2.4*pi*-3);
pred_3 = net(3);
pred_ne_3 = net(-3);
disp(y3)
disp(ne_y3)
disp(pred_3)
disp(pred_ne_3)


