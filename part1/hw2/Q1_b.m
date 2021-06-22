clear;
clc;
syms x y;
w = rand(2,1);
f=(1-x)^2+100*(y-x^2)^2;
H=hessian(f,[x,y])
mat_x = [w(1)];
mat_y = [w(2)];
train_value=(1-w(1))^2+100*(w(2)-w(1)^2)^2;
mat_value=[train_value]

for epoch=1:100000
    grad_w = [ 400*w(1)^3+2*w(1)-400*w(1)*w(2)-2 ; 
              200*w(2)-200*w(1)^2];
    h=subs(H,[x,y],[w(1),w(2)]);
    w = w - inv(h) * grad_w;
    mat_x(end+1)=w(1);
    mat_y(end+1)=w(2);
    train_value=(1-w(1))^2+100*(w(2)-w(1)^2)^2;
    mat_value(end+1)=train_value;
    if abs(train_value)<0.00001
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
plot(0:epoch,mat_value,'Linewidth',2)
title('trajectory of function value');
xlabel('number of iterations');
ylabel('value');
grid on 