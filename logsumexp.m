
function log_sum = logsumexp(log_vector)
   % Ensure the input is a column vector
    log_vector = log_vector(:); 
    
    % Find the maximum log-probability
    max_log = max(log_vector);
    
    % If the maximum log-probability is -Inf (i.e., all inputs are -Inf),
    % then return -Inf since the log of zero probabilities is -Inf.
    if isinf(max_log)
        log_sum = -Inf;
        return;
    end
    
    % Compute the log-sum-exp by subtracting the max log-probability for numerical stability,
    % summing the exponentiated differences, and then taking the log
    log_sum = max_log + log(sum(exp(log_vector - max_log)));
end