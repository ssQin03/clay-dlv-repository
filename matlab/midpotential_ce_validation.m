clc;
clear;

% 主程序，可以对给定无量纲表面电荷密度，给出拟合解和数值解的对比情况，2025年7月21日

ED0 = 19;  %表面无量纲电荷密度，σ/sqrt(2*n0*ybsl*kt)，实际上是表面的无量纲场强
D_values = 0.1:0.1:10; %颗粒间无量纲间距，从0.2到1，步长为0.2
results = zeros(size(D_values)); % 初始化结果数组

results_fit=zeros(size(D_values)); % 拟合解
for i = 1:length(D_values)
    D = D_values(i);
    results(i) = simplified_solver(ED0, D);
    results_fit(i)= fitted_solution(ED0,D);

end

% 输出结果

hold on;
plot(D_values,results,'o');
plot(D_values,results_fit,'+'); 
legend({'numerical','fitted'}); 
xlabel('无量纲间距D','FontWeight','bold');
ylabel('无量纲中面电势Ud','FontWeight','bold');
grid on;
legend show;




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

function u = fitted_solution(eb, D)

    if eb<=10
        if eb<=1.6
            a=0.249*eb^(-1.3298)+5.5979;
        else
            a=0.9739*log(eb)+4.7257;
        end
        b=0.8936*eb^(-0.6444)+0.5535;
        c=0.6743*exp(0.0069*eb)-0.348*exp(-0.8048*eb);
    elseif eb<=1000
        if eb<=500
            a=0.9739*log(eb)+4.7257;
        else
            a=-60.0941*eb^(-0.7154)+11.339;
        end
        b=1.0267*exp(0.0001*eb)-0.2608*exp(-0.0115*eb);
        c=0.1191*exp(-0.0126*eb)+0.6136;
    else
        a=-159.3808*eb^(-0.8861)+11.2594;
        b=-15.0559*eb^(-0.8903)+1.1213;
        c=5.0068*eb^(-0.8648)+0.5739;
    end
    u=a*exp(-b*D^c);
end