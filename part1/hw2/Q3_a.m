% A0225480J mod(80, 1) + 1 = 3 Group 3 Glasses Wearing Detection
clear;
clc;
%% Input Data
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

%% Plot label distribution
Wearing_train = sum(label_train);
Wearing_test = sum(label_test);
label_distribution = [Wearing_train, num_train-Wearing_train; Wearing_test, num_test-Wearing_test];
X = categorical({'Training', 'Testing'});
X = reordercats(X,{'Training', 'Testing'});
b = bar(X,label_distribution,'stacked');
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
legend('wear glasses','no glasses');
title('distribution of training set and test set')