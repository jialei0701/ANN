%****************************
%目的：BP神经网络预测部分，计算出损失函数
%时间：2019/4/13
%程序员：Jarvis
%****************************
function Predict_YY = Jarvis_NN_Predict(Predict_x,Wo,Bo,Wl,Bl)
Data_size=size(Predict_x,1);%预测集的数量
Num_X=size(Predict_x,2);%Num_X=3
Num_L=size(Bl,1);%Num_L=50
Num_Y=size(Bo,1);%Num_Y=6
L=zeros(Num_L,1);%50个隐含层节点
Y=zeros(Num_Y,Data_size);%6个输出节点(作为结果输出)
    for kk=1:Data_size
        for ii=1:Num_L
            Midd=0;
            for jj=1:Num_X
                Midd=Midd+Predict_x(kk,jj)*Wl(jj,ii);%求加权和
            end
            L(ii,1)=Fun_Sigmoid(Midd,Bl(ii,1));%求隐含层的结果
        end
        for ii=1:Num_Y
            Midd=0;
            for jj=1:Num_L
                Midd=Midd+L(jj)*Wo(jj,ii);%求加权和
            end
            Y(ii,kk)=Fun_Sigmoid(Midd,Bo(ii,1));%求输出层的结果
        end
    end
Predict_YY=Y';%预测结果
end