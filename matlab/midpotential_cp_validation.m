clc;
clear;


% 主程序，可以对给定表面电势，给出拟合解和数值解的对比情况，2025年7月20日
kt=4.14e-21;% 玻尔兹曼常数乘以温度，假设温度为300开尔文，单位焦耳
n0=0.01*6.02e23*1000; %离子浓度，对应于0.01摩尔每升
ybsl=7.08e-10; %介电常数，单位库伦/(伏特*米)
e0=1.6e-19; %单电荷，单位库伦
v0=1; %化合价
h_dbl= sqrt(ybsl*kt/2/n0/(v0*e0)^2)*1e9; %双电层厚度，单位nm

b0 = 0.2;  %表面无量纲电势
D_values = 0.02:0.02:10; %颗粒间无量纲间距，从0.02到1，步长为0.01
results = zeros(size(D_values)); % 初始化结果数组
dis_2 = zeros(size(D_values));poten_mid = zeros(size(D_values));
results_ana_1=zeros(size(D_values)); % 近似理论解-1
results_ana_2=zeros(size(D_values)); % 近似理论解-2
results_ana_3=zeros(size(D_values)); % 近似理论解-3
results_fit=zeros(size(D_values)); % 拟合解
for i = 1:length(D_values)
    D = D_values(i);
    results(i) = simplified_solver(b0, D);
    results_fit(i)= fitted_solution(b0,D);
    results_ana_1(i)=2 * log(2 * pi / (D + 4 * exp(-0.5 * b0)));
    results_ana_2(i)=4 * log((1 + tanh(0.25 * b0) * exp(-0.5 * D)) / ...
                        (1 - tanh(0.25 * b0) * exp(-0.5 * D)));
    results_ana_3(i)=log(8 * (sqrt(1 + 0.25 * D * D * exp(b0)) - 1) / (D * D));
    %转变为有量纲量
    dis_2(i) = D * h_dbl; %有量纲颗粒间距
    poten_mid(i) = results(i)*kt/(v0*e0);

end

% 输出结果

hold on;
plot(D_values,results,'o');
% plot(D_values,results_ana_1,'-');
% plot(D_values,results_ana_2,'*');
% plot(D_values,results_ana_3,'+');
plot(D_values,results_fit,'+');
% legend({'numerical','analy-3','analy-4','analy-5'}); 
legend({'numerical','fitted'}); 
xlabel('无量纲间距D','FontWeight','bold');
ylabel('无量纲中面电势Ud','FontWeight','bold');
grid on;
legend show;




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

function u = fitted_solution(b0, D)

    if b0<=1
        a=1.0329*b0;
        b=0.0155*b0^2+0.0002*b0+0.1705;
        c=-0.0422*b0^2-0.0042*b0+1.4769;
    elseif b0<=8
        a=1.0093*b0+0.0704;
        b=-0.0012*b0^3+0.0201*b0^2-0.0069*b0+0.1745;
        c=0.0009*b0^3-0.0052*b0^2-0.1191*b0+1.5648;
    else
        a=8.0;
        b=0.0052*b0^2-0.1151*b0+1.3852;
        c=-0.0043*b0^2+0.0969*b0+0.2249;
    end
    u=a*exp(-b*D^c);
end