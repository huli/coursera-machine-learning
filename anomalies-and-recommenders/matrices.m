A = ones(5,3);
B = ones(3, 5) * 2;
C = A * B;


R = [1 0 0 0 0
        1 0 0 0 0
        1 0 1 0 0
        1 0 0 0 0
        0 0 0 0 1
        ];
        
total = 0;
for i = 1:5
    for j = 1:5
        if (R(i,j) == 1)
          total = total + C(i, j);
        end
    end
end

total

sum(sum(C .* R))

sum(sum((A * B) .* R))

C = A * B; total = sum(sum(C(R==1)))

C = (A * B) * R; total = sum(C(:))

sum(sum(A(R==1) * B(R==1)))