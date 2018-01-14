function [d] = euclideanDistance (u, v)
  d = sqrt(sum((u - v) .** 2));
endfunction
