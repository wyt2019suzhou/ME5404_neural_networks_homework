clear;
clc;
train_x=-2:0.05:2;
train_y=1.2*sin(pi*train_x)-cos(2.4*pi*train_x);
train_num=size(train_x,2);
test_x=-2:0.01:2;
test_y=1.2*sin(pi*test_x)-cos(2.4*pi*test_x);
x_cell = num2cell(train_x,1);
y_cell = num2cell(train_y,1);
%% net
net = feedforwardnet(10);
net = configure(net,x_cell,y_cell );
net.divideFcn = 'dividetrain';
net.trainParam.lr=0.001;
net.performFcn = 'mse';
max_epoch=1000;
pred_train = net(train_x);
pred_test = net(test_x);
mat_perf_train=[perform(net,pred_train,train_y)];
mat_perf_test=[perform(net,pred_test,test_y)];
%% sequential train
for j=1:max_epoch
    idx = randperm(train_num);
    net = adapt(net,x_cell(idx), y_cell(idx));
    pred_train = net(train_x);
    pred_test = net(test_x);
    perf_train = perform(net,pred_train,train_y);
    perf_test = perform(net,pred_test,test_y);
    mat_perf_train(end+1)=perf_train;
    mat_perf_test(end+1)=perf_test;
    if perf_train<0.001
        break
    end
end

%% result
figure(1)
% title('N=10 Comparison of test set output and desired result');
xlabel('x');
ylabel('y');
plot(test_x,test_y,'Linewidth',3);
hold on
plot(test_x,pred_test,'Linewidth',1);
legend('desired result','test set output');
title('N=100');
hold off
% figure(2)
% title('N=10 performance on train set and test set ');
% xlabel('number of epoch');
% ylabel('mse');
% plot(0:size(mat_perf_train,2)-1,mat_perf_train,'Linewidth',3);
% hold on
% plot(0:size(mat_perf_train,2)-1,mat_perf_test,'Linewidth',1);
% legend('mse of train set','mse of test set');
% hold off
y3=1.2*sin(pi*3)-cos(2.4*pi*3);
ne_y3=1.2*sin(pi*-3)-cos(2.4*pi*-3);
pred_3 = net(3);
pred_ne_3 = net(-3);
disp(y3)
disp(ne_y3)
disp(pred_3)
disp(pred_ne_3)