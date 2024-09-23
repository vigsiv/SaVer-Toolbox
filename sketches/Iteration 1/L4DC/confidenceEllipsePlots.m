clear all; close all;

% load in output data from python script
load('Dataout.mat');
load('CFout.mat');

fs = 20;
ms = 20;
skip = 50;

figure(2); hold on; grid on;
xlabel('$x$','Interpreter','latex','FontSize',fs);
ylabel('$y$','Interpreter','latex','FontSize',fs);
set(gca, 'FontSize', fs);
box on;
scatter(Dataout(1,1:skip:end), Dataout(2,1:skip:end), ms, 'ok');

dx = 0.004;
CFout1 = CFout(1,:)';
CFout2 = CFout(2,:)';
gradCFout1 = gradient(CFout1, dx);
gradgradCFout1 = gradient(gradCFout1, dx);
gradCFout2 = gradient(CFout2, dx);
gradgradCFout2 = gradient(gradCFout2, dx);
% meanOut1 = real(1i * gradCFout1(5000))
% meanOut2 = real(1i * gradCFout2(5000))
trueMeanOut1 = mean(Dataout(1,:));
trueMeanOut2 = mean(Dataout(2,:));
% stdOut1 = imag(sqrt(-gradgradCFout1(5000) - meanOut1^2))
% stdOut2 = imag(sqrt(-gradgradCFout2(5000) - meanOut2^2))
trueStdOut1 = std(Dataout(1,:));
trueStdOut2 = std(Dataout(2,:));

h1 = error_ellipse('mu', mean(Dataout'), 'C', cov(Dataout'), 'conf', 0.9973, 'style', '-.r');
h2 = error_ellipse('mu', [trueMeanOut1; trueMeanOut2], 'C', blkdiag(trueStdOut1^2, trueStdOut2^2), 'conf', 0.99973, 'style', 'b');
h1.LineWidth = 2;
h2.LineWidth = 2;

