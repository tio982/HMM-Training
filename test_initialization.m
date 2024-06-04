
function test_initialization()
    num_states = 8;
    feature_dimension = 13;

    % Test means and covariance initialization
    [test_means, test_covariances] = initialise_gaussian_parameters(num_states, feature_dimension);

    disp('Test Means:');
    disp(test_means);

    disp('Test Covariances for State 1:');
    disp(test_covariances(:, :, 1));
end

function [means, covariances] = initialise_gaussian_parameters(num_states, feature_dimension)
    means = randn(num_states, feature_dimension);
    covariances = zeros(feature_dimension, feature_dimension, num_states);
    for i = 1:num_states
        cov_matrix = eye(feature_dimension) * 1e-3; % Small value for covariance
        covariances(:, :, i) = cov_matrix;
    end
end

% Call the test function
% test_initialization();