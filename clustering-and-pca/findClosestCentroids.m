function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

for i = 1:rows(X)
  
  smallest_distance = 1000;
  nearest_centroid = 0;
  for c = 1:K
   
    % Compute distance of current centroid
    current_distance  = euclideanDistance(X(i, :), centroids(c, :));
    
    % Compare to smallest distance until now
    if(current_distance < smallest_distance)
      smallest_distance = current_distance;
      nearest_centroid = c;
    end
  end
  
  % Assign smallest distance to result
  idx(i) = nearest_centroid;
  
end

