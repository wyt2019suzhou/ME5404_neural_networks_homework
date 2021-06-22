clear
clc
%% load data
load('train.mat')
load('test.mat')
%% Preprocessing data
%Standardization
mu = mean(train_data,2);
sigma = std(train_data,1,2); 
norm_train = (train_data - mu)./sigma;
norm_test = (test_data - mu)./sigma;
%% P loop
for P = [1,2,3,4,5]
%% Mercer condition check
gram_m = (norm_train'*norm_train+1).^P;
eigenvalues = eig(gram_m);
flag = true;
if min(eigenvalues) <-1e-4
    flag = false;
    fprintf('P = %d this kernel candidate is not admissible\n',P)
end
%% Training
if flag == true
    % Set the training parameters
    A = [];
    b = [];
    Aeq = train_label';
    Beq = 0;
    [f_dim,s_dim]=size(norm_train);
    lb = zeros(s_dim,1);
    C = 10e6; 
    ub=ones(s_dim,1)*C;
    f=-ones(s_dim,1);
    x0 = [];
    H_sign = train_label*train_label';
    H = (norm_train'*norm_train + 1).^P.*H_sign;
    options = optimset('MaxIter',200);
    Alpha = quadprog(H,f,A,b,Aeq,Beq,lb,ub,x0,options);
    idx = find(Alpha>1e-4);
    % Calculate disciminant parameters
    more_K = sum(Alpha.*train_label.*(norm_train'*norm_train+1).^P,1); 
    bo = mean(train_label(idx) - more_K(idx));
    % performance evaluate
    acc_train = Acc(Alpha,bo,norm_train,train_label,norm_train,train_label,P);
    fprintf('acc_train:%.2f%% when P=%d C=%.1f\n ',acc_train*100,P,C)
    acc_test = Acc(Alpha,bo,norm_train,train_label,norm_test,test_label,P);
    fprintf('acc_test:%.2f%% when P=%d C=%.1f\n ',acc_test*100,P,C)
end
end
%% functions
function accuracy = Acc(alpha,bo,train_data,train_label,data,label,P)
    more_K =sum(alpha.*train_label.*(train_data'*data+1).^P,1)'; % (num_sppport x s_dim)
    pred_label = sign(more_K + bo);
    accuracy = mean(pred_label == label,'all');
end