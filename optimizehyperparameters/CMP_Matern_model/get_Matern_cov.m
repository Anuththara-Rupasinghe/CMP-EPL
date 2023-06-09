function K_matrix = get_Matern_cov(xsamp,noise_len,noise_rho,order)

    distance_matrix = abs(xsamp(:)-xsamp(:)');

    switch order
        case 0
            K_matrix = noise_rho*exp(-distance_matrix/noise_len);
        case 1
            K_matrix = noise_rho*(1 + sqrt(3)*distance_matrix/noise_len).*exp(-sqrt(3)*distance_matrix/noise_len);
        case 2 
            K_matrix = noise_rho*(1 + sqrt(5)*distance_matrix/noise_len + 5*distance_matrix.^2/(3*noise_len^2)).*exp(-sqrt(5)*distance_matrix/noise_len);
        otherwise
            fprintf("Valid polynomial orders are only 0, 1, or 2!")
            return
    end

end