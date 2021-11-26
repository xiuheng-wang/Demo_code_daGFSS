% Script for distributed and adaptive GFSS. 
% Algorithm 2 in "Distributed Change Detection in Streaming Graph Signals".
%
% 2021/04
% Implemented by
% Xiuheng Wang.
% The ARMA filter code is downloaded from:
% https://andreasloukas.blog/2017/09/04/what-is-the-most-efficient-graph-filter-tchebychev-vs-arma/

clear;clc;
close all;
addpath('Tchebychev vs ARMA');

t = 200;
tc = 100;
gamma = 0.06;
% learning rates:
lambda = 0.3;
LAMBDA = 0.6;
mode = 1; % 0 or 1: full-connected or not full-connected
threshold = 0.1;
% signal model:
sigma = 1;
scale_factor = 10; % delta / sigma
%% Produce degree matrix, normalized Laplacian matrix and its shifted version
W = load('W.mat').W;
N = size(W,1);
if mode
    W(W < threshold)= 0;
    % Normalize
    for i = 1:N
        W(i,:)=W(i,:)/sum(W(i,:));
    end
    for i = 1:N
        for j = i:N
           W(j,i) = W(i,j);
        end
    end
end
D = zeros(N);
for i = 1:N
   D(i,i) = sum(W(:,i)); 
end
L = D^(-0.5) * (D - W) * D^(-0.5);

%% Generate simulated graph signals
M = 10 + 10 * rand([N, 1]);
Y = M + sigma * randn(N, t); 
delta = sigma * scale_factor;
Delta = delta + randn([4, 1]); % standard deviation = 1
M2 = M(1: 4) + Delta; 
Y2 = M2 + sigma * randn(4, t - tc);
Y(1: 4, tc + 1: end) = Y2;
figure;
for i = 1:N
    plot(Y(i, :));
    hold on;
end

%% Sorted Eigenvalues
[U,mu] = eig(L); % Factorize the Laplacian
mu = diag(mu);
[mu,ind] = sort(mu);

%% Estimate response function with an ARMA filter
response = min(1, sqrt(gamma ./ mu));
Kb = 4;
Ka = 4;
[b, a, rARMA, ~] = agsp_design_ARMA( mu, response, Kb, Ka);
figure;
plot(response);
hold on;
plot(rARMA);

%% Using the ARMA filter
GY = zeros(size(Y));
for i = 1:t
        gy = agsp_filter_ARMA( L, b, a, Y(:, i), 10, 0);
        GY(:, i) = gy(:, end);
end
figure;
for i = 1:N
    plot(GY(i, :));
    hold on;
end

%% Adaptive strategy form NEWMA
VT_1 = zeros(N, t+1);
VT_2 = VT_1;
for i = 2:t+1
    VT_1(:, i) = (1 - lambda) * VT_1(:, i-1) + lambda * GY(:, i-1);
    VT_2(:, i) = (1 - LAMBDA) * VT_2(:, i-1) + LAMBDA * GY(:, i-1);
end
VT_1(:, 1) = []; VT_2(:, 1) = []; 
VT = VT_2 - VT_1;

figure;
for i = 1:N
    plot(VT(i, :));
    hold on;
end

%% Detect changes in a distributed manner
tdaGFSS = zeros(size(Y));
if mode == 1
    W(W >  0)= 1;
    for i = 1:N
        tdaGFSS(i, :) = W(i, :) * VT / sum(W(i, :));
    end
    else
    for i = 1:N
        tdaGFSS(i, :) = W(i, :) * VT;
    end
end