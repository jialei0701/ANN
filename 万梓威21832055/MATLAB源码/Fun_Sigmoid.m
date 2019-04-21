%****************************
%目的：S型激活函数
%时间：2019/4/13
%程序员：Jarvis
%****************************
function FunS_Value= Fun_Sigmoid(X,Bias)
    if X+Bias<-10   %下饱和
        FunS_Value=-1;
    elseif X+Bias>10   %上饱和
        FunS_Value=1;
    else
        FunS_Value=2/(1+2.718^(-2*(X+Bias)))-1;
    end
end