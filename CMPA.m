%set up initial values
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

%generate vectors
V = [-1.95:0.01331:0.7];
I = zeros(1,200);
Ivar = zeros(1,200);


for ( i = 1:200)
    var = 0.4.*rand(1) + 0.8;
    Vp = V(i);
    I(i) = Is*(exp(48*Vp) - 1) + Gp*Vp - Ib*(exp(-48*(Vp+Vb)) - 1);
    Ivar(i) = I(i)*var;
end

%poly fitting
p_I4 = polyfit(V,I,4);
p_Ivar4 = polyfit(V,Ivar,4);
p_I8 = polyfit(V,I,8);
p_Ivar8 = polyfit(V,Ivar,8);

eq_I4 = polyval(p_I4,V);
eq_Ivar4 = polyval(p_Ivar4,V);
eq_I8 = polyval(p_I8,V);
eq_Ivar8 = polyval(p_Ivar8,V);

%plot results

%poly fit
figure(1)
plot(V,I,'b');
hold on
title('Plot of Data (polyfit, nonrandom)');
plot(V,eq_I4,'r');
plot(V,eq_I8,'g');
xlabel('V');
ylabel('I');
legend('I','poly4','poly8');

figure(2)
plot(V,Ivar,'b');
hold on
title('Plot of Data (polyfit, random)');
plot(V,eq_Ivar4,'r');
plot(V,eq_Ivar8,'g');
xlabel('V');
ylabel('I');
legend('Ivar','poly4','poly8');

%semilog fit
figure(3)
semilogy(V,abs(I),'b');
hold on
semilogy(V,abs(eq_I4),'r');
semilogy(V,abs(eq_I8),'g');
title('Plot of Data (nonrandom, polyfit, semilog axis)');
xlabel('V');
ylabel('I');
legend('I','poly4', 'poly8');

figure(4)
semilogy(V,abs(Ivar),'b');
hold on
semilogy(V,abs(eq_Ivar4),'r');
semilogy(V,abs(eq_Ivar8),'g');
title('Plot of Data (random, polyfit, semilog axis)');
xlabel('V');
ylabel('Ivar');
legend('Ivar','poly4', 'poly8');


%fitted curves
%2 fitted parameters
fo2 = fittype('A.*(exp(48*x) - 1) + Gp.*x -C*(exp(-48*(x+Vb))-1)');
ff2 = fit(V.',Ivar.',fo2);
If2 = ff2(V);

%3 fitted parameters
fo3 = fittype('A.*(exp(48*x) - 1) + B.*x -C*(exp(-48*(x+Vb))-1)');
ff3 = fit(V.',Ivar.',fo3);
If3 = ff3(V);

%all 4 parameters
fo4 = fittype('A.*(exp(48*x) - 1) + B.*x -C*(exp(-48*(x+D))-1)');
ff4 = fit(V.',Ivar.',fo4);
If4 = ff4(V);

%plot
figure(5)
plot(V,Ivar,'b')
hold on
plot(V,If2,'r');
plot(V,If3,'g');
plot(V,If4,'y');
title('Nonlinear fit');
legend('Ivar','2fit', '3fit','4fit');

figure(6)
semilogy(V,abs(Ivar),'b')
hold on
semilogy(V,abs(If2),'r');
semilogy(V,abs(If3),'g');
semilogy(V,abs(If4),'y');
title('Nonlinear fit, log scale');
legend('Ivar','2fit', '3fit','4fit');

%neural net
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

%neural net plot
figure(7)
plot(V,Ivar,'b')
hold on
plot(V,Inn,'r');
title('Neural fit');
legend('Ivar','neural fit');

figure(8)
semilogy(V,abs(Ivar),'b')
hold on
semilogy(V,abs(Inn),'r');
title('Neural net fit, log scale');
legend('Ivar','neural fit');
