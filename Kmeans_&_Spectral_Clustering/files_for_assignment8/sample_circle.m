function [data ,label] = sample_circle( num_cluster, points_per_cluster )
% Function to sample 2-D circle-shaped clusters
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
points_per_cluster=points_per_cluster(:);

data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);
idx = 1;
bandwidth = 0.1;

for k = 1 : num_cluster
    theta = 2 * pi * rand(points_per_cluster(k), 1);
    rho = k + randn(points_per_cluster(k), 1) * bandwidth;
    [x, y] = pol2cart(theta, rho);
    data(idx:idx+points_per_cluster(k)-1,:) = [x, y];
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
end
