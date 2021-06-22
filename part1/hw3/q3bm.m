clc
clear
close all;
rng(520)
%% Train data
x = randn(800, 2);
s2 = sum(x.^2, 2);
trainX = (x.*repmat(1*(gammainc(s2/2, 1).^(1/2))./sqrt(s2), 1, 2))';
n_train=size(trainX,2);
M=6;
N=6;
T=600;
neurons = rand(2, M, N);
sigma0 = sqrt(M^2 + N^2) / 2;
lr0 = 0.1;
constant=T/log(sigma0);
%%  train som
for epoch = 1:T
    lr = lr0*exp(-epoch/T);
    sigma = sigma0*exp(-epoch/constant);
    for i = 1:n_train
        dis = squeeze(sum((trainX(:,i) - neurons).^2, 1))';
        [~, winner] = min(dis, [], 'all', 'linear'); 
        row = ceil(winner/M);
        col = mod(winner, N);
        if col == 0
            col = N;
        end
        d0 = ([1:M] - row).^2; 
        d1 = ([1:N] - col).^2; 
        d = d1' + d0; 
        h0 = exp(-d ./ (2*sigma^2));
        h = permute(repmat(h0, [1 1 2]), [3 2 1]);
        neurons = neurons + lr*h.*(trainX(:,i) - neurons);
    end
end

hold on; 
grid on; 
axis equal;
plot(trainX(1,:),trainX(2,:),'+r');
for i = 1:M
    for j = 1:N
        if i+1 <= M
            plot([neurons(1,i,j),neurons(1,i+1,j)], [neurons(2,i,j),neurons(2,i+1,j)], 'bo-');
        end
        if j+1 <= N
            plot([neurons(1,i,j),neurons(1,i,j+1)], [neurons(2,i,j),neurons(2,i,j+1)], 'bo-');
        end
    end
end
xlabel('x'); 
ylabel('y');
legend('Train set', 'SOM weight');
title('SOM function approximation');

