% Script for adaptive GFSS.
% Algorithm 1 in "Distributed Change Detection in Streaming Graph Signals".
%
% 2021/03
% Implemented by
% Xiuheng Wang.
% 

clear;clc;
close all;
t = 200;
tc = 100;
gamma = 0.06;
% learning rates:
lambda = 0.3;
LAMBDA = 0.6;
mode = 1; % 0 or 1: full-connected or not full-connected
threshold = 0.1;
%% Produce degree matrix and normalized Laplacian matrix
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
% L = D^(-1) * (D - W);
% L = D - W;

%% Eigenvectors associated smallest K eigenvalues
[U,mu] = eig(L); % Factorize the Laplacian
mu = diag(mu);
[mu,ind] = sort(mu);
U = U(:, ind);
figure;
plot(mu, '*');
K = 4; % 4 eigenvalues is small
figure;
for i = 1:K
    plot(U(:, i));
    hold on;
end

%% Generate simulated graph signals
M = 10 + 10 * rand([N,1]);
Y = M + 0.2 * randn(N, t); 
M2 = 2 + 6 * rand([4,1]);
Y2 = M2  +  0.2 * randn(4, t - tc); 
Y(1:4, tc + 1: end) = Y2; % first cluster

figure;
for i = 1:N
    plot(Y(i, :));
    hold on;
end

%% Using the GFSS filter
GY = zeros(size(Y));
for i = 1:t
    for j = 2:N
        gy = min(1, sqrt(gamma / mu(j))) * (U(:, j)' * Y(:, i)) * U(:, j);
        GY(:, i) = GY(:, i) + gy;
    end
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

VT = VT_1 - VT_2;
taGFSS = zeros(t, 1);
for i = 1:t
    taGFSS(i) = norm(VT(:, i));
end
figure;
plot(taGFSS);