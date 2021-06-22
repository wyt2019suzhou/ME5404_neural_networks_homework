
clear;
clc;
%% Input and resize Data
groupID = 3;
train_dir = dir('Face Database/TrainImages/*.jpg');
test_dir = dir('Face Database/TestImages/*.jpg');
label_train_dir = dir('Face Database/TrainImages/*.att');
label_test_dir = dir('Face Database/TestImages/*.att');
num_train = size(label_train_dir, 1);
num_test = size(label_test_dir, 1);
img_train = zeros(10201, num_train);
img_test = zeros(10201, num_test);
label_train = zeros(1, num_train);
label_test = zeros(1, num_test);

for i = 1:num_train
    img_name = train_dir(i).name;
    label_name = label_train_dir(i).name;
    img = imread(['Face Database/TrainImages/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_train(:, i) = img_v(1:10201);
    label = load(['Face Database/TrainImages/',  label_name]);
    label_train(i) = label(groupID);
end
for j = 1:num_test
    img_name = test_dir(j).name;
    label_name = label_test_dir(j).name;
    img = imread(['Face Database/TestImages/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_test(:, j) = img_v(1:10201);
    label = load(['Face Database/TestImages/', label_name]);
    label_test(j) = label(groupID);
end


image_cell = num2cell(img_train,1);
lable_cell = num2cell(label_train,1);

%% multi-layer sequence train
n_hidden=50;
net = patternnet(n_hidden,'traincgb')
net = configure(net,image_cell,lable_cell);
net.performFcn = 'mse';
net.divideFcn ='dividetrain';
max_epoch=100;
net.trainParam.lr=0.0001;
pred_train = net(img_train);
pred_test = net(img_test);
mat_perf_train=[perform(net,pred_train,label_train)];
mat_perf_test=[perform(net,pred_test,label_test)];
train_num=size(img_train,2);
for j=1:max_epoch
    idx = randperm(train_num);
    net = adapt(net,image_cell(idx), lable_cell(idx));
    pred_train = net(img_train);
    pred_test = net(img_test);
    perf_train = perform(net,pred_train,label_train);
    perf_test = perform(net,pred_test,label_test);
    mat_perf_train(end+1)=perf_train;
    mat_perf_test(end+1)=perf_test;
    disp(j);
    if perf_train<0.0001
        break
    end
end

% [net, tr]  = train(net, all_img, all_label);

%% anlysis
perf_train = 1 - mean(abs(label_train-pred_train));
perf_test = 1 - mean(abs(label_test-pred_test));
figure(1)
plot(perf_train, 'linewidth', 1.5);
hold on;
plot(perf_test, 'linewidth', 1.5);
xlabel('number of epoch'); 
ylabel('mse');
legend('train mse', 'test mse');
title('mutiple layer perceptron ');