
function models = initialise_hmm_models(num_words, num_states, feature_dimension, aii, aii_plus_1)
    models = repmat(struct('A', [], 'means', [], 'covariances', []), num_words, 1);
    
    for i = 1:num_words
        % Initialize transition probability matrix A
        A = zeros(num_states); 

        % Setting transition probabilities
        for s = 1:num_states-1
            A(s, s) = aii;
            A(s, s + 1) = aii_plus_1;
        end
        A(num_states, num_states) = 1;  % Last state has self-transition probability of 1

        % Normalize each row to sum up to 1
        row_sums = sum(A, 2);
        A_normalised = bsxfun(@rdivide, A, row_sums);

        models(i).A = A_normalised; % Use normalised probabilities
        [models(i).means, models(i).covariances] = initialise_gaussian_parameters(num_states, feature_dimension);
    end
end

function [means, covariances] = initialise_gaussian_parameters(num_states, feature_dimension)
    % Initialize Gaussian parameters for each state
    means = randn(num_states, feature_dimension) * 0.5;
    
    % Set initial covariance matrices with a small positive diagonal
    small_value = 1;
    covariances = repmat(eye(feature_dimension) * small_value, [1, 1, num_states]);
end

function result = log_minus(log_a, log_b)
    % Log subtraction: log(a - b) in a numerically stable way
    result = log_a + log(1 - exp(log_b - log_a));
end