clc;
clear;


% 主程序  2025-07-21
% 恒表面电荷密度条件，先求出0.02~0.3电荷密度、0.02~10德拜长度间距下的各中面电势
% 再用各表面电势条件下的中面电势解，用a*exp(-b*(x^c))来拟合，获得相应的拟合系数
kt=4.14e-21;% 玻尔兹曼常数乘以温度，假设温度为300开尔文，单位焦耳
n0=0.01*6.02e23*1000; %离子浓度，对应于0.01摩尔每升
ybsl=7.08e-10; %介电常数，单位库伦/(伏特*米)
e0=1.6e-19; %单电荷，单位库伦
v0=1; %化合价
h_dbl= sqrt(ybsl*kt/2/n0/(v0*e0)^2)*1e9; %双电层厚度，单位nm（后面用不到，用nm是为了方便查看理解）

ED= 0.2:0.2:10;  %无量纲表面电荷密度，σ/sqrt(2*n0*ybsl*kt)，实际上是表面的无量纲场强
ED1= 20:10:100; ED2= 150:50:300;ED3= 400:100:1000;ED4= 1500:500:5000; %为适应其分段非线性强度
ED0=[ED,ED1,ED2,ED3,ED4];
D_values = 0.02:0.02:10; %颗粒间无量纲间距，从0.02到1，步长为0.02

poten_mid = zeros(size(ED0,2),size(D_values,2));% 初始化结果数组

%求解耗时，将计算结果保存为poten_mid_ce_full.mat，后续调用
%{
for j=1:length(ED0)
    for i = 1:length(D_values)
        poten_mid(j,i) = simplified_solver(ED0(j), D_values(i));
    end
    hold on;
    plot(D_values,poten_mid(j,:),'o');
    xlabel('无量纲间距D','FontWeight','bold');
    ylabel('无量纲中面电势Ud','FontWeight','bold');
    grid on;
    legend show;
end
%}

load poten_mid_ce_full.mat;

D_values =transpose(D_values);poten_mid=transpose(poten_mid);
a_1=zeros(size(ED0,2),1);b_1=zeros(size(ED0,2),1);c_1=zeros(size(ED0,2),1);
R2_1=zeros(size(ED0,2),1);
% 开始非线性拟合
for i=1:length(ED0)
    xData=D_values;yData=poten_mid(:,i);
    % 设置 fittype 和选项。
    ft = fittype( 'a*exp(-b*(x^c))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Lower = [0 0 0];
    opts.StartPoint = [0.2 0.9 0.3];
    opts.Upper = [12 12 12];
    % 对数据进行模型拟合。
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a_1(i)=fitresult.a;b_1(i)=fitresult.b;c_1(i)=fitresult.c;
    R2_1(i)=gof.rsquare;
end


function u = simplified_solver(eb, D)
    % 定义被积函数
    integrand = @(y, a0) 1./sqrt(2*cosh(y) - 2*cosh(a0));
    
    % 定义求解 u 的函数

    func = @(a0) integral(@(y) integrand(y, a0), acosh(0.5*eb^2+cosh(a0)), a0) + D/2;
    
    % 使用 fzero 求解方程 func(u) = 0
    a0_initial_guess = 2*asinh(0.5*eb)*exp(-D/2); % 初始猜测值,2*asinh(sqrt(0.5*eb))是由Grahame方程确定
    options = optimset('TolFun', 1e-7, 'MaxIter', 800);
    u_solution = fzero(func, a0_initial_guess, options);
    a0_initial_guess_0 =a0_initial_guess;
    k=0;
    while isnan(u_solution)
        k=k+1;
        if k<10
            a0_initial_guess = 0.8 * a0_initial_guess; 
            u_solution = fzero(func, a0_initial_guess, options);
        else
            a0_initial_guess_0 = 1.5 * a0_initial_guess_0; 
            u_solution = fzero(func, a0_initial_guess_0, options);
        end
    end
    
    % 返回结果
    u = u_solution;
end