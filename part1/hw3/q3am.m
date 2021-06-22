clc
clear
close all;
rng(520)
%% Train data
x = linspace(-pi,pi,400);
trainX = [x;2*sin(x)]; 
n_train=size(trainX,2);
M=1;
N=36;
T=600;
neurons = rand(M,N);
sigma0 = sqrt(M^2+N^2)/2;
lr0 = 0.1;
constant=T/log(sigma0);
%%  train som
for epoch = 1:T
    lr = lr0*exp(-epoch/T);
    sigma = sigma0*exp(-epoch/constant);
    for i = 1:n_train
        dis = sum((trainX(:,i) - neurons).^2,1);
        [~,winner] = min(dis,[],2);
        d = abs([1:36]-winner);
        h = exp(-d.^2/(2*sigma^2));
        % Update
        neurons = neurons + lr*h.*(trainX(:,i) - neurons);
    end
end
hold on
plot(trainX(1,:),trainX(2,:),'+r');
plot(neurons(1,:),neurons(2,:),'bo-');
xlabel('x');
ylabel('y');
legend('Train set', 'SOM weight');
title('SOM function approximation');
hold off

