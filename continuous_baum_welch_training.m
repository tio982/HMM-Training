
function [updated_A, updated_means, updated_covariances, updated_log_matrix_b, B] = continuous_baum_welch_training(initial_A, initial_means, initial_covariances, observed_sequence, num_iterations, feature_dimension)
    A = exp(initial_A); % Convert log probabilities to normalised probabilities
    means = initial_means;
    covariances = initial_covariances;
    B = []; % Initialize the B matrix

    for iteration = 1:num_iterations
        [log_matrix_b, matrix_b] = compute_log_b_matrix(means, covariances, observed_sequence);
        [gamma, log_alpha, log_beta, log_matrix_b] = baum_welch_e_step(A, means, covariances, observed_sequence, feature_dimension);
        [A, means, covariances, B] = reestimate_parameters(A, means, covariances, observed_sequence, gamma, log_alpha, log_beta, log_matrix_b, matrix_b, feature_dimension); % Include matrix_b in the function call
    end

    updated_A = log(A); % Convert probabilities back to log probabilities
    updated_means = means;
    updated_covariances = covariances;
    updated_log_matrix_b = log_matrix_b; % Save the updated log_matrix_b
end



function [gamma, log_alpha, log_beta, log_matrix_b] = baum_welch_e_step(A, means, covariances, observed_sequence, feature_dimension)
    T = size(observed_sequence, 1);  % The length of the observed sequence
    num_states = size(A, 1);         % The number of states in the model

    % Call compute_log_b_matrix without feature_dimension
    log_matrix_b = compute_log_b_matrix(means, covariances, observed_sequence);

    [log_alpha, C] = forwards_algo(A, log_matrix_b, T, num_states);
    log_beta = backwards_algo(A, log_matrix_b, T, num_states, C);

    gamma = compute_gamma(log_alpha, log_beta, T, num_states);
end


function [log_matrix_b, matrix_b] = compute_log_b_matrix(means, covariances, observed_sequence)
    num_states = size(means, 1);
    T = size(observed_sequence, 1);
    log_matrix_b = zeros(num_states, T); % Log probability matrix
    matrix_b = zeros(num_states, T);     % Normal probability matrix

    for j = 1:num_states
        for t = 1:T
            pdf_value = multivariateGaussianPDF(observed_sequence(t, :), means(j, :), covariances(:, :, j));
            log_matrix_b(j, t) = log(pdf_value + 1e-150); % Log probability
            matrix_b(j, t) = pdf_value;                  % Regular probability
        end
    end
    % Normalizing the probabilities for each time step
    matrix_b = bsxfun(@rdivide, matrix_b, sum(matrix_b, 1) + 1e-150);
end



function pdf_value = multivariateGaussianPDF(x, mu, Sigma)
    % Calculate the dimensions
    n = size(x, 2); % Assuming 'x' is a row vector for a single observation
    
    % Regularize the covariance matrix to avoid singularity issues
    regularisation_value = 1e-6;
    sigma_reg = Sigma + eye(size(Sigma)) * regularisation_value;
    
    % Calculate the pseudo-inverse and pseudo-determinant for numerical stability
    invSigma = pinv(sigma_reg);
    detSigma = pdet(sigma_reg);
    
    % Calculate the coefficient
    coef = 1 / ((2 * pi)^(n/2) * sqrt(detSigma));
    
    % Calculate the exponent
    xDiff = (x - mu)';
    expo = -0.5 * sum((xDiff' * invSigma) .* xDiff', 2);
    
    % Compute the PDF value
    pdf_value = coef * exp(expo);
end

function p_det = pdet(Sigma)
    % Computes the pseudo-determinant of a matrix that might have
    % very small eigenvalues, treating them as zero.
    eigenvalues = eig(Sigma);
    p_det = prod(eigenvalues(eigenvalues > 1e-300));
    if p_det == 0
        p_det = 1e-300; % Set a lower bound to avoid log(0)
    end
end

% Example of scaling in the forward algorithm
function [log_alpha, C] = forwards_algo(A, log_matrix_b, T, num_states)
    log_alpha = -Inf(num_states, T);
    log_alpha(:, 1) = log_matrix_b(:, 1);
    C = zeros(1, T); % Scaling factors

    % Compute the scaling factor for the first column
    C(1) = 1 / sum(exp(log_alpha(:, 1)));
    log_alpha(:, 1) = log_alpha(:, 1) + log(C(1));

    for t = 2:T
    for i = 1:num_states
        temp = log_alpha(:, t-1) + log(A(:, i));
        log_alpha(i, t) = log_sum_exp(temp) + log_matrix_b(i, t);
    end
    C(t) = 1 / sum(exp(log_alpha(:, t)));
    if C(t) == 0
        error('Scaling factor C(%d) is zero, leading to numerical instability.', t);
    end
    log_alpha(:, t) = log_alpha(:, t) + log(C(t));
    end
end


function log_beta = backwards_algo(A, log_matrix_b, T, num_states, C)
    log_beta = -Inf(num_states, T);
    log_beta(:, T) = log(C(T));  % Scaling the last column with the final scaling factor

    log_beta(:, T) = log(C(T)); % Scaling the last column with the final scaling factor
if C(T) == 0
    error('Final scaling factor C(T) is zero, leading to numerical instability.');
end

for t = T-1:-1:1
    for i = 1:num_states
        temp = log(A(i, :))' + log_matrix_b(:, t+1) + log_beta(:, t+1);  % ensure temp is a column vector
        log_beta(i, t) = log_sum_exp(temp);  % this should now return a scalar
    end
    log_beta(:, t) = log_beta(:, t) + log(C(t));  % apply the scaling factor
end
end

function gamma = compute_gamma(log_alpha, log_beta, T, num_states)
    log_gamma = bsxfun(@plus, log_alpha, log_beta);
    normalized_log_gamma = normalize_log_probs(log_gamma);
    gamma = exp(normalized_log_gamma);
end


function normalised_log_probs = normalize_log_probs(log_probs)
    max_log_prob = max(log_probs, [], 2);
    log_probs = bsxfun(@minus, log_probs, max_log_prob);
    log_probs_sum = log_sum_exp(log_probs);
    normalised_log_probs = bsxfun(@minus, log_probs, log_probs_sum);
end



function [new_A, new_means, new_covariances, B] = reestimate_parameters(A, means, covariances, observed_sequence, gamma, log_alpha, log_beta, log_matrix_b, matrix_b, feature_dimension)
    num_states = size(A, 1);
    T = size(observed_sequence, 1);
    new_A = zeros(size(A));
    new_means = zeros(size(means));
    new_covariances = zeros(feature_dimension, feature_dimension, num_states);
    for i = 1:num_states
        for j = 1:num_states
            log_xi_sum = log_sum_exp(bsxfun(@plus, bsxfun(@plus, log_alpha(i, 1:T-1), log(A(i, j))), bsxfun(@plus, log_beta(j, 2:T), log_matrix_b(j, 2:T))));
            new_A(i, j) = exp(log_xi_sum) / sum(gamma(i, 1:T-1));
        end
    end

    for j = 1:num_states
        gamma_sum = sum(gamma(j, :));
        new_means(j, :) = sum(bsxfun(@times, observed_sequence, gamma(j, :)'), 1) / gamma_sum;
        for t = 1:T
            diff = observed_sequence(t, :) - new_means(j, :);
            new_covariances(:, :, j) = new_covariances(:, :, j) + (gamma(j, t) * (diff' * diff));
        end
        new_covariances(:, :, j) = new_covariances(:, :, j) / gamma_sum;
        new_covariances(:, :, j) = new_covariances(:, :, j) + eye(feature_dimension) * 0.5;
    end
     B = matrix_b; % Store the B matrix
end


function log_sum = log_sum_exp(log_vector)
    max_log = max(log_vector);
    scaled_log_vector = log_vector - max_log;
    sum_of_exps = sum(exp(scaled_log_vector));
    log_sum = max_log + log(sum_of_exps);
end

