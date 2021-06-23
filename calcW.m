function W = calcW(a_true,res_ols)
    % Create the W matrix from the residual of the OLS
    N = length(res_ols);
    idx_big = find(a_true > 0.475);
    idx_small = find(a_true <= 0.475);

    i_big = length(idx_big);
    i_small = length(idx_small);

    big_cm = zeros(1,i_big);
    small_cm = zeros(1,i_small);
    for j = 1:i_big
        big_cm(j) = res_ols(idx_big(i_big));
    end

    for j = 1:i_small
        small_cm(j) = res_ols(idx_small(i_small));
    end

    rho_small = std(small_cm)^2;
    rho_big   = std(big_cm)^2;

    W = diag(ones(1,N));
    W = W.*eye(N);

    counter = 1;
    for j = idx_big
        W(j,j) = rho_big;
        counter = counter + 1;
    end
    counter = 1;
    for j = idx_small
        W(j,j) = rho_small;
        counter = counter + 1;
    end
end