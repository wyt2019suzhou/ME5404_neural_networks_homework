clear
clc
%% load data
load('train.mat')
load('test.mat')
%% Preprocessing data
%no normalize
% norm_train = train_data;
% norm_test = test_data;

%Standardization
mu = mean(train_data,2);
sigma = std(train_data,1,2); 
norm_train = (train_data - mu)./sigma;
norm_test = (test_data - mu)./sigma;
% Minmax
% [norm_train,PS] = mapminmax(train_data);
% norm_test=mapminmax('apply',test_data,PS);

% %unit length
% norm_train  = train_data /norm(train_data );
% norm_test  = test_data /norm(test_data );
%% Mercer condition check
gram_m = norm_train'*norm_train;
eigenvalues = eig(gram_m);
flag = true;
if min(eigenvalues) < -1e-4
    flag = false;
    fprintf('this kernel candidate is not admissible')
end
%% Training
if flag == true
    % Set the training parameters
    % Hard margin: +inf(in theory)/10e6 (in practice)
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
    H = norm_train'*norm_train.*H_sign;
    options = optimset('MaxIter',200);
    Alpha = quadprog(H,f,A,b,Aeq,Beq,lb,ub,x0,options);
    % select support vectors
    idx = find(Alpha>1e-4);
    % Calculate disciminant parameters
    wo = sum(Alpha'.*train_label'.*norm_train,2);
    bo=mean(1./train_label(idx) - norm_train(:,idx)'*wo);
    % performance evaluate
    acc_train = Acc(wo,bo,norm_train,train_label);
    fprintf('acc_train:%.2f%% when C=%d\n',acc_train*100,C)
    acc_test = Acc(wo,bo,norm_test,test_label);
    fprintf('acc_test:%.2f%% when C=%d\n',acc_test*100,C)

end
%% evaluation functions
function accuracy = Acc(w,b,data,label)
    pred_label = sign(w'*data+b)';
    accuracy = mean(pred_label == label,'all');
end