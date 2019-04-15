%****************************
%目的：BP神经网络的继续训练，网络结构3-20-6
%时间：2019/4/13
%程序员：Jarvis
%****************************
clc
close all
MaxEpoch=10000; %最大训练量
Goal=0.0001; %目标方差
LearningRatio1=0.05;%输出层学习率
LearningRatio2=0.05;%输出层学习率
Mse_record=zeros(MaxEpoch,1);%损失函数值记录

%************* 训练网络(损失函数为MES) ***************
for Epoch=1:MaxEpoch
    NN_Y=Jarvis_NN_Predict(Predict_x,NN_Wo,NN_Bo,NN_Wl,NN_Bl);%预测输出
    
    Midd=0;
    for ii=1:Data_size_Predict
        for jj=1:NN_Num_Y
           Midd=Midd+(NN_Y(ii,jj)-Predict_y(ii,jj))^2; %计算平方和
        end
    end
    Mse_record(Epoch,1)=Midd/Data_size_Predict/NN_Num_Y;%计算MSE
    disp(['Epoch = ' num2str(Epoch) ', MSE = ' num2str(Mse_record(Epoch,1))]);
        
    if Mse_record(Epoch,1)>Goal %如果没达到要求
        [NN_Wo,NN_Bo,NN_Wl,NN_Bl]=Jarvis_NN_Train(Train_x,Train_y,NN_Wo,NN_Bo,NN_Wl,NN_Bl,LearningRatio1,LearningRatio2);%训练网络
    else
        break
    end
end

%************* 绘制结果 ***************
figure(1);
plot(Mse_record);
xlabel('Epoch');
ylabel('MSE');
title('Loss Function');
axis([1 MaxEpoch 0 0.001]);

Scale=100;%绘制细分100份
v1=linspace(-1,1,Scale+1);%细分100
v2=linspace(-1,1,Scale+1);%细分100
Z1=zeros(Scale+1);%Fx
Z2=zeros(Scale+1);%Fy
Z3=zeros(Scale+1);%Fz
Z4=zeros(Scale+1);%Tx
Z5=zeros(Scale+1);%Ty
Z6=zeros(Scale+1);%Tz
[X, Y]=meshgrid(v1,v2);
Test_y=zeros(1,NN_Num_Y);
for ii= 0:Scale
    for jj=0:Scale
        Test_y=Jarvis_NN_Predict([ii*2/Scale-1,jj*2/Scale-1,1],NN_Wo,NN_Bo,NN_Wl,NN_Bl);%预测输出
        Z1(jj+1,ii+1)=Test_y(1,1);
        Z2(jj+1,ii+1)=Test_y(1,2);
        Z3(jj+1,ii+1)=Test_y(1,3);
        Z4(jj+1,ii+1)=Test_y(1,4);
        Z5(jj+1,ii+1)=Test_y(1,5);
        Z6(jj+1,ii+1)=Test_y(1,6);
    end
end
figure(2);
mesh(45*X, 45*Y, 125*Z1);
hold on
for ii=1:size(Train_x,1)
    if Train_x(ii,3)==1
        scatter3(45*Train_x(ii,1),45*Train_x(ii,2),125*Train_y(ii,1),'k') 
    end
end
xlabel('α / (°)');
ylabel('β / (°)');
zlabel('Fx / N');
title('Fx value (B = 0mm)');
axis([-45 45 -45 45 -125 125]);
figure(3);
mesh(45*X, 45*Y, 125*Z2);
hold on
for ii=1:size(Train_x,1)
    if Train_x(ii,3)==1
        scatter3(45*Train_x(ii,1),45*Train_x(ii,2),125*Train_y(ii,2),'k') 
    end
end
xlabel('α/(°)');
ylabel('β/(°)');
zlabel('Fy / N');
title('Fy value (B = 0mm)');
figure(4);
mesh(45*X, 45*Y, 500*Z3);
hold on
for ii=1:size(Train_x,1)
    if Train_x(ii,3)==1
        scatter3(45*Train_x(ii,1),45*Train_x(ii,2),500*Train_y(ii,3),'k') 
    end
end
xlabel('α/(°)');
ylabel('β/(°)');
zlabel('Fx / N');
title('Fz value (B = 0mm)');
figure(5);
mesh(45*X, 45*Y, 3*Z4);
hold on
for ii=1:size(Train_x,1)
    if Train_x(ii,3)==1
        scatter3(45*Train_x(ii,1),45*Train_x(ii,2),3*Train_y(ii,4),'k') 
    end
end
xlabel('α/(°)');
ylabel('β/(°)');
zlabel('Tx / Nm');
title('Tx value (B = 0mm)');
figure(6);
mesh(45*X, 45*Y, 3*Z5);
hold on
for ii=1:size(Train_x,1)
    if Train_x(ii,3)==1
        scatter3(45*Train_x(ii,1),45*Train_x(ii,2),3*Train_y(ii,5),'k') 
    end
end
xlabel('α/(°)');
ylabel('β/(°)');
zlabel('Ty / Nm');
title('Ty value (B = 0mm)');
figure(7);
mesh(45*X, 45*Y, 3*Z6);
hold on
for ii=1:size(Train_x,1)
    if Train_x(ii,3)==1
        scatter3(45*Train_x(ii,1),45*Train_x(ii,2),3*Train_y(ii,6),'k') 
    end
end
xlabel('α/(°)');
ylabel('β/(°)');
zlabel('Tz / Nm');
title('Tz value (B = 0mm)');
