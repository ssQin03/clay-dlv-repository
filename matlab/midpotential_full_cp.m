clc;
clear;


% 主程序  2025-07-20
% 恒表面电势条件，先求出0.2~10无量纲表面电势、0.02~10德拜长度间距下的各中面电势
% 再用各表面电势条件下的中面电势解，用a*exp(-b*(x^c))来拟合，获得相应的拟合系数
kt=4.14e-21;% 玻尔兹曼常数乘以温度，假设温度为300开尔文，单位焦耳
n0=0.01*6.02e23*1000; %离子浓度，对应于0.01摩尔每升
ybsl=7.08e-10; %介电常数，单位库伦/(伏特*米)
e0=1.6e-19; %单电荷，单位库伦
v0=1; %化合价
h_dbl= sqrt(ybsl*kt/2/n0/(v0*e0)^2)*1e9; %双电层厚度，单位nm

b0 = 0.2:0.2:10;  %表面无量纲电势
D_values = 0.02:0.02:10; %颗粒间无量纲间距，从0.02到1，步长为0.01

poten_mid = zeros(size(b0,2),size(D_values,2));% 初始化结果数组

for j=1:length(b0)
    for i = 1:length(D_values)
        poten_mid(j,i) = simplified_solver(b0(j), D_values(i));
    end
    hold on;
    plot(D_values,poten_mid(j,:),'o');
    xlabel('无量纲间距D','FontWeight','bold');
    ylabel('无量纲中面电势Ud','FontWeight','bold');
    grid on;
    legend show;
end

D_values =transpose(D_values);poten_mid=transpose(poten_mid);
a_1=zeros(size(b0,2),1);b_1=zeros(size(b0,2),1);c_1=zeros(size(b0,2),1);
R2_1=zeros(size(b0,2),1);
% 开始非线性拟合
for i=1:length(b0)
    xData=D_values;yData=poten_mid(:,i);
    % 设置 fittype 和选项。
    ft = fittype( 'a*exp(-b*(x^c))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Lower = [0 0 0];
    opts.StartPoint = [0.2 0.9 0.3];
    opts.Upper = [8 8 8];
    % 对数据进行模型拟合。
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a_1(i)=fitresult.a;b_1(i)=fitresult.b;c_1(i)=fitresult.c;
    R2_1(i)=gof.rsquare;
end


function u = simplified_solver(b0, D)
    % 定义被积函数
    integrand = @(y, a0) 1./sqrt(2*cosh(y) - 2*cosh(a0));
    
    % 定义求解 u 的函数
    func = @(a0) integral(@(y) integrand(y, a0), b0, a0) + D/2;
    
    % 使用 fzero 求解方程 func(u) = 0
    a0_initial_guess = 2*b0*exp(D/2)/(1+exp(D)); % 初始猜测值
    options = optimset('TolFun', 1e-8, 'MaxIter', 800);
    u_solution = fzero(func, a0_initial_guess, options);
    while isnan(u_solution)
        a0_initial_guess = 0.9 * a0_initial_guess; %上面初始猜测值，对于较大表面电势情况下，需要调小，以保证解可以收敛
        u_solution = fzero(func, a0_initial_guess, options);
    end
    
    % 返回结果
    u = u_solution;
end