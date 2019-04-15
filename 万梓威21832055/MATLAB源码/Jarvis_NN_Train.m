%****************************
%目的：BP神经网络训练部分，更新权重与阈值
%时间：2019/4/13
%程序员：Jarvis
%****************************
function [Wo,Bo,Wl,Bl] = Jarvis_NN_Train(Train_x,Train_y,Wo,Bo,Wl,Bl,Ratio1,Ratio2)
Data_size=size(Train_x,1);%预测集的数量
Num_X=size(Train_x,2);%Num_X=3
Num_L=size(Bl,1);%Num_L=50
Num_Y=size(Bo,1);%Num_Y=6
X=zeros(Num_X,1);%3个输入节点
L=zeros(Num_L,1);%50个隐含层节点
Y=zeros(Num_Y,1);%6个输出节点
dL=zeros(Num_L,1);%50个隐含层误差
dY=zeros(Num_Y,1);%6个输出误差

for oo=1:Data_size
    for ii=1:Num_L
        Midd=0;
        for jj=1:Num_X
        Midd=Midd+Train_x(oo,jj)*Wl(jj,ii);%求加权和
        end
        L(ii,1)=Fun_Sigmoid(Midd,Bl(ii,1));%求隐含层的结果
    end
    for ii=1:Num_Y
        Midd=0;
        for jj=1:Num_L
        Midd=Midd+L(jj,1)*Wo(jj,ii);%求加权和
        end
        Y(ii,1)=Fun_Sigmoid(Midd,Bo(ii,1));%求输出层的结果
    end
    for ii=1:Num_Y
        dY(ii,1)=(Train_y(oo,ii)-Y(ii,1))*0.5*(1-Y(ii,1)^2);%输出层误差
    end
    for ii=1:Num_L
        Midd=0;
        for jj=1:Num_Y
            Midd=Midd+Wo(ii,jj)*dY(jj,1);
        end
        dL(ii,1)=Midd*0.5*(1-L(ii,1)^2);%隐含层误差
    end
    for ii=1:Num_Y
        for jj=1:Num_L
            Wo(jj,ii)=Wo(jj,ii)+Ratio2*dY(ii,1)*L(jj,1);%权重调节
        end
        Bo(ii,1)=Bo(ii,1)+Ratio2*dY(ii,1);%阈值调节
    end
    for ii=1:Num_L
        for jj=1:Num_X
            Wl(jj,ii)=Wl(jj,ii)+Ratio1*dL(ii,1)*X(jj,1);%权重调节
        end
        Bl(ii,1)=Bl(ii,1)+Ratio1*dL(ii,1);%阈值调节
    end
end
end