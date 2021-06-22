clc;
clear;
close all;
%% label map
load('characters10.mat');
rng(520)
load('neurons')
M = 10;
N = 10;
group1Idx = find(train_label==1); 
group1Data = train_data(group1Idx, :);
group2Idx = find(train_label==2); 
group2Data = train_data(group2Idx, :);
group3Idx = find(train_label==3); 
group3Data = train_data(group3Idx, :);
group4Idx = find(train_label==4); 
group4Data = train_data(group4Idx, :);
group5Idx = find(train_label==5); 
group5Data = train_data(group5Idx, :);
group6Idx = find(train_label==6); 
group6Data = train_data(group6Idx, :);
group7Idx = find(train_label==7); 
group7Data = train_data(group7Idx, :);
group9Idx = find(train_label==9); 
group9Data = train_data(group9Idx, :);
maplable=zeros(M,N);
hold on; 
axis equal;
for i = 1:M
    for j = 1:N
        subplot(M, N, M*(i-1)+j);
        imshow(reshape(neurons(:,i,j), [28 28]), [0 255], 'InitialMagnification', 'fit');
        dist1 = mean(squeeze(sum((double(group1Data)' - neurons(:,i,j)).^2, 1))','all');
        dist2 = mean(squeeze(sum((double(group2Data)' - neurons(:,i,j)).^2, 1))','all');
        dist3 = mean(squeeze(sum((double(group3Data)' - neurons(:,i,j)).^2, 1))','all');
        dist4 = mean(squeeze(sum((double(group4Data)' - neurons(:,i,j)).^2, 1))','all');
        dist5 = mean(squeeze(sum((double(group5Data)' - neurons(:,i,j)).^2, 1))','all');
        dist6 = mean(squeeze(sum((double(group6Data)' - neurons(:,i,j)).^2, 1))','all');
        dist7 = mean(squeeze(sum((double(group7Data)' - neurons(:,i,j)).^2, 1))','all');
        dist9 = mean(squeeze(sum((double(group9Data)' - neurons(:,i,j)).^2, 1))','all');
        dist=[dist1 dist2 dist3 dist4 dist5 dist6 dist7 10000000000000 dist9];
        [~, winner] = min(dist, [], 'all', 'linear'); 
        maplable(i,j)=winner;
        title(sprintf('Label:%d',winner))
    end
end
suptitle('SOM conceptual map');
hold off;