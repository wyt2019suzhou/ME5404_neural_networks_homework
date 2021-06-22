clear 
clc
x= rand;
y = rand;
lr = 0.5;
mat_x = [x];
mat_y = [y];
mat_grad_x =[400*x^3+2*x-400*x*y-2]
mat_grad_y =[200*y-200*x^2]
train_value=(x-1)^2+100*(y-x^2)^2;
mat_value=[train_value];
for epoch=1:100000
    grad_x = 400*x^3+2*x-400*x*y-2 ;     
    grad_y = 200*y-200*x^2 ;
    x = x - lr*grad_x;
    y = y - lr*grad_y;
    mat_x(end+1)=x;
    mat_y(end+1)=y;
    mat_grad_x(end+1)=grad_x;
    mat_grad_y(end+1)=grad_y;
    train_value=(1-x)^2+100*(y-x^2)^2;
    mat_value(end+1)=train_value;
    if abs(train_value) < 0.00001
        break
    end
end

figure(1)
plot(mat_x,mat_y,'Linewidth',2)
title('trajectory of (x,y)');
xlabel('x');
ylabel('y');
grid on
figure(2)
plot(0:epoch,mat_value,'Linewidth',2);
title('trajectory of function value');
xlabel('number of iterations');
ylabel('value');
grid on 
figure(3)
plot(0:epoch,mat_grad_x,'Linewidth',2);
title('trajectory of x gradient');
xlabel('number of iterations');
ylabel('x gradient');
grid on 
figure(4)
plot(0:epoch,mat_grad_y,'Linewidth',2);
title('trajectory of y gradient');
xlabel('number of iterations');
ylabel('y gradient');
grid on 
disp(x)
disp(y)
disp(train_value)
disp(epoch)

