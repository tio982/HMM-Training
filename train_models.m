
function [trained_models, all_emission_probs] = train_models(audio_files, AUDIOSET_FOLDER, num_words, num_states, feature_dimension, aii, aii_plus_1, num_iterations)
% Preallocate cell array for storing MFCCs
    mfcc_features = cell(num_words, 1);

    % Process each audio file
    for i = 1:numel(audio_files)
        current_fname = audio_files{i};
        audio_file_path = fullfile(AUDIOSET_FOLDER, current_fname);

        if ~isfile(audio_file_path)
            warning('File does not exist: %s', current_fname);
            continue;
        end

        temp_mfcc = extract_mfcc(audio_file_path, feature_dimension);
        if isempty(temp_mfcc)
            warning('MFCC extraction returned empty for: %s', current_fname);
            continue;
        end

        word_index = extract_word_index(current_fname) + 1; % Adjust for MATLAB's 1-based indexing
        if word_index < 1 || word_index > num_words
            warning('Invalid word index: %d in file %s', word_index, current_fname);
            continue;
        end

        % Append features to the corresponding word's cell array
        mfcc_features{word_index} = [mfcc_features{word_index}; temp_mfcc];
    end

    % Normalise features outside the training loop
    % Assuming mfcc_features is already defined and is a cell array
normalised_mfcc_features = cell(size(mfcc_features));  % Initialise the cell array

for i = 1:numel(mfcc_features)
    if isempty(mfcc_features{i})
        % Handle empty MFCC feature sets, which might be placeholders for errors
        normalised_mfcc_features{i} = [];
    else
        % Normalise the MFCC features for each audio file
        normalised_mfcc_features{i} = normalise_mfcc_features(mfcc_features{i}, feature_dimension);
    end
end

    % Initialise HMM models
    models = initialise_hmm_models(num_words, num_states, feature_dimension, aii, aii_plus_1);

    % Train HMM Models using Baum-Welch algorithm
    trained_models = repmat(struct('A', [], 'means', [], 'covariances', []), num_words, 1);
    all_emission_probs = cell(num_words, 1); % Cell array to store emission probabilities for each word
    for i = 1:num_words
        if isempty(normalised_mfcc_features{i})
            disp(['No data for word index ', num2str(i), '. Skipping model training.']);
            continue;
        end

        disp(['Training model for word index ', num2str(i)]);
[trained_A, trained_means, trained_covariances, emission_probs, trained_B] = continuous_baum_welch_training(models(i).A, models(i).means, models(i).covariances, normalised_mfcc_features{i}, num_iterations, feature_dimension);

        % Store trained models and emission probabilities
        trained_models(i).A = trained_A;
        trained_models(i).means = trained_means;
        trained_models(i).covariances = trained_covariances;
        all_emission_probs{i} = emission_probs; % Store emission probabilities for the current word
        trained_models(i).B = trained_B;
    end
end


function word_index = extract_word_index(filename)
    % Extract word index from filename
    pattern = '_w(\d+)';  % Pattern to find '_w' followed by digits
    token = regexp(filename, pattern, 'tokens');
    if isempty(token) || isempty(token{1})
        warning('Filename does not contain a valid word index: %s', filename);
        word_index = []; % Set word_index to empty to indicate an error
    else
        word_index = str2double(token{1}{1});
        if isnan(word_index)
            warning('Failed to convert word index to a number for filename: %s', filename);
            word_index = []; % Set word_index to empty to indicate an error
        end
    end
end



