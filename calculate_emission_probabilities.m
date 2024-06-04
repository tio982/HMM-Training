
function B = calculate_emission_probabilities(observed_sequence, means, covariances)
    % Number of states and observations
    num_states = size(means, 1);
    T = size(observed_sequence, 1);

    % Initialize emission probability matrix B
    B = zeros(num_states, T);

    % Calculate emission probabilities for each state and time step
    for j = 1:num_states
        for t = 1:T
            B(j, t) = mvnpdf(observed_sequence(t, :), means(j, :), covariances(:, :, j));
        end
    end
end