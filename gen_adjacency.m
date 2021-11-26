% Script for generating adjacency matrix.
% This code borrows heavily from: https://github.com/jermwatt/spectral-clustering-demo
%
% 2021/03
% Implemented by
% Xiuheng Wang.
% 

clear;clc;
close all;
format long;
%% Define vertices on the graph
stop = 0;
axis([0 10 0 10])    % Set viewing axes
x = [];
y = [];
sigma = 2;
while stop == 0
    hold on
    [a,b]=ginput(1);
    if sum(size(a)) > 0 && (a > 0 && a < 10 && b > 0 && b < 10)
        x = [x ; a];
        y = [y ; b];
        scatter(a,b,'fill','b')
    else
        stop = 1;
    end  
end
saveas(gcf, 'example.jpg');
close(gcf)
pts = [x y];
n = size(pts,1);

%% Produce weighted adjacency matrix 
W = zeros(n);
for i = 1:size(pts,1)
    for j = i:size(pts,1)
        if i ~= j
            dist = exp( - norm(pts(i,:) - pts(j,:))^2 / (2 * sigma * sigma)); % Gaussian similarity function
            W(i,j) = dist;
            W(j,i) = dist;
        end
    end
end
% Normalize
for i = 1:size(pts,1)
    W(i,:)=W(i,:)/sum(W(i,:));
end
for i = 1:size(pts,1)
    for j = i:size(pts,1)
       W(j,i) = W(i,j);
    end
end
