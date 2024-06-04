
function logp = multivariate_gaussian(x, mu, Sigma)
    n = length(mu); % Number of dimensions
    Sigma = Sigma + eye(size(Sigma)) * 1; % Regularization to avoid singularity
    Sigma_det = det(Sigma);
    if Sigma_det <= 0
        error('Covariance matrix must be positive definite.');
    end
    Sigma_inv = inv(Sigma);
    diff_x_mu = x - mu;
    logp = -0.5 * n * log(2 * pi) - 0.5 * log(Sigma_det) - 0.5 * sum((diff_x_mu * Sigma_inv) .* diff_x_mu, 2);
end