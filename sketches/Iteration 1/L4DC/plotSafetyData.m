clear all; close all;
% load in data from python script

load('Data2.mat');
load('CDF2.mat');
plotSafety(Data2, 0.05, CDF2);


function plotSafety(data, p, cdf)

close all;
figure; hold on; 

N = length(data);
grey = [0.7 0.7 0.7];
gridCF = linspace(-200,200,10000);

meanOut = mean(data);
stdOut = std(data);

lbConf = meanOut - stdOut / sqrt(3*p);
ubConf = meanOut + stdOut / sqrt(3*p);

gridConf = linspace(lbConf, ubConf, 1000)';
tickMark = linspace(-0.01, 0.01, 1000)';

s = scatter(data, zeros(N, 1), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', grey);
distFromMean = sqrt((data - meanOut).^2);
s.AlphaData = distFromMean;
s.MarkerFaceAlpha = 'flat';
s.SizeData = 200;
quiver(0, 0, 2, 0, 0, '-b', 'LineWidth', 2);
plot(gridConf, zeros(1000, 1), '-r', 'LineWidth', 2);
plot(ones(1000, 1) * lbConf, tickMark, '-r', 'LineWidth', 5);
plot(ones(1000, 1) * ubConf, tickMark, '-r', 'LineWidth', 5);
plot(ones(1000, 1) * min(data), tickMark, '-k', 'LineWidth', 5);
plot(ones(1000, 1) * max(data), tickMark, '-k', 'LineWidth', 5);
plot(ones(1000, 1) * 0, tickMark, '-b', 'LineWidth', 5);
set(gca,'FontSize',24);
yticks([]);

figure; hold on; 
[ycdf, xcdf] = cdfcalc(data);
xccdf = xcdf;
yccdf = 1 - ycdf(1 : end - 1);
plot(xccdf, yccdf, '-r', 'LineWidth', 2);
plot(gridCF,1 - cdf(1,:),'.-k','MarkerSize',12);
xlim([min(data)-0.1,max(data)+0.1]);
set(gca,'FontSize',24);
xlabel('$x$','Interpreter','latex','FontSize',20);
ylabel('$\bar{\Phi}_{\mathbf{y}}$','Interpreter','latex','FontSize',20);




end
