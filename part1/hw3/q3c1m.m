clc;
close all;
clear;
load('characters10.mat');
rng(520)
T = 1000;
lr0 = 0.1;
M = 10;
N = 10;
trainIdx = find(train_label~=0 & train_label~=8); 
trainLabel = train_label(trainIdx);
trainData = train_data(trainIdx, :);
n_train = length(trainLabel);
neurons =  rand(size(trainData,2),M,N);
sigma0 = sqrt(M^2+N^2)/2;
constant=T/log(sigma0);
for epoch = 1:T
    lr = lr0*exp(-epoch/T);
    sigma = sigma0*exp(-epoch/constant);
    for i = 1:n_train
        dist = squeeze(sum((double(trainData(i,:))' - neurons).^2, 1))';
        [~, winner] = min(dist, [], 'all', 'linear'); 
        row = ceil(winner/M);
        col = mod(winner, N);
        if col == 0
            col = N;
        end
        d0 = ([1:M] - row).^2; 
        d1 = ([1:N] - col).^2; 
        d = d1' + d0; 
        h0 = exp(-d ./ (2*sigma^2));
        h = permute(repmat(h0, [1 1 size(trainData,2)]), [3 2 1]);
        neurons = neurons + lr*h.*(double(trainData(i,:))' - neurons);
    end
end
%% label and plot
maplable=zeros(M,N);
hold on; 
axis equal;
for i = 1:M
    for j = 1:N
        subplot(M, N, M*(i-1)+j);
        imshow(reshape(neurons(:,i,j), [28 28]), [0 255], 'InitialMagnification', 'fit');
        dist = squeeze(sum((double(trainData)' - neurons(:,i,j)).^2, 1))';
        [~, winner] = min(dist, [], 'all', 'linear'); 
        maplable(i,j)=trainLabel(winner);
        title(sprintf('Label:%d',trainLabel(winner)))
    end
end
suptitle('SOM conceptual map');
hold off;

