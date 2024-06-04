
function log_beta = backwards_algo(log_A, log_matrix_b, observed_sequence, log_scaling_factor)
    % Validate input dimensions
    [num_states, num_transitions] = size(log_A);
    if num_states ~= num_transitions || num_states ~= size(log_matrix_b, 1)
        error('Dimension mismatch in transition and emission matrices.');
    end

    sequence_length = length(observed_sequence);
    if sequence_length < 1
        error('Empty observed sequence.');
    end

    % Initialize log_beta
    log_beta = -Inf(num_states, sequence_length);
    log_beta(:, sequence_length) = log_scaling_factor(sequence_length);

    % Backward iteration (vectorized)
    for t = sequence_length-1:-1:1
        for i = 1:num_states
            log_beta(i, t) = logsumexp(log_A(i, :)' + log_matrix_b(:, observed_sequence(t+1)) + log_beta(:, t+1));
        end
        log_beta(:, t) = log_beta(:, t) + log_scaling_factor(t); % Scale in log space
    end
end

function log_sum = logsumexp(log_vector)
    % Helper function for the log-sum-exp trick
    max_log = max(log_vector);
    if isinf(max_log)
        log_sum = max_log;
    else
        log_sum = max_log + log(sum(exp(log_vector - max_log)));
    end
end






