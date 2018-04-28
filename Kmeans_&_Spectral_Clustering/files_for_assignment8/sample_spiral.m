function [data, label] = sample_spiral(num_cluster, points_per_cluster)
% Function to sample 2-D spiral-shaped clusters
% Input:
% num_cluster: the number of clusters 
% points_per_cluster: a vector of [num_cluster] numbers, each specify the
% number of points in each cluster 
% Output:
% data: sampled data points. Each row is a data point;
% label: ground truth label for each data points.
%
% EC 503: Learning from Data
% Instructor: Prakash Ishwar
% Assignment 8, Problem 8.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 0
  num_cluster = 2;
end
if nargin == 1
   points_per_cluster = 500*ones(num_cluster,1);
end
delta = 2*pi/num_cluster;
bandwidth = 0.1;
points_per_cluster=points_per_cluster(:);

data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);

idx = 1;

for k = 1 : num_cluster
    w = rand(points_per_cluster(k),1);
    data(idx:idx+points_per_cluster(k)-1,1) = (4 * w + 1) .* cos(2 * pi * w + (k-1)*delta) + randn(points_per_cluster(k),1) * bandwidth;
    data(idx:idx+points_per_cluster(k)-1,2) = (4 * w + 1) .* sin(2 * pi * w + (k-1)*delta) + randn(points_per_cluster(k),1) * bandwidth;
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
end

% points_per_cluster = 500;
% bandwidth = 0.1;
% 
% data = zeros([points_per_cluster, 2]);
% for k = 1 : points_per_cluster
%   w = k / points_per_cluster;
%   data(k,1) = (4 * w + 1) * cos(2 * pi * w) + randn(1) * bandwidth;
%   data(k,2) = (4 * w + 1) * sin(2 * pi * w) + randn(1) * bandwidth;
% end
% 
% data = [data; -data];
