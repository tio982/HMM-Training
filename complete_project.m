
% Folder paths
AUDIOSET_FOLDER = '/Users/tb/Documents/MATLAB/SAAR Part 2 Assignement/EEEM030cw2_DevelopmentSet';
OUT_FOLDER = '/Users/tb/Documents/MATLAB/SAAR Part 2 Assignement/audio_processing_results';
OUT_SUBFOLDER = 'mfcc_extracted_data';

num_words = 13; % Number of words or categories
num_states = 10; % Number of states in HMM
feature_dimension = 13; % Dimensionality of features
aii = 0.8; % Probability of staying in the same state
aii_plus_1 = 0.2; % Probability of moving to the next state
num_iterations = 15; % Number of iterations for training

% Path to all audio files
all_audio_files = dir(fullfile(AUDIOSET_FOLDER, '*.mp3'));
all_audio_files = {all_audio_files.name}; % Extracting names

% Split index for training-testing split
num_files = length(all_audio_files);
num_train_files = round(num_files * 0.9); % 90% for training

% Split data into training and testing sets
train_audio_files = all_audio_files(1:num_train_files);
test_audio_files = all_audio_files(num_train_files+1:end);

% Extract and process MFCC features from audio files
train_mfcc_features = feature_extraction(train_audio_files, AUDIOSET_FOLDER, feature_dimension);

% Normalise MFCC features for each audio file
% Ensure train_mfcc_features is a cell array with content
if isempty(train_mfcc_features) || ~iscell(train_mfcc_features)
    error('train_mfcc_features must be a non-empty cell array.');
end

% Initialise the normalised features cell array
normalised_train_mfcc_features = cell(size(train_mfcc_features));

% Loop through each element in the cell array for normalisation
for i = 1:length(train_mfcc_features)
    if isempty(train_mfcc_features{i})
        normalised_train_mfcc_features{i} = [];
    else
        normalised_train_mfcc_features{i} = normalise_mfcc_features(train_mfcc_features{i}, feature_dimension);
    end
end

% Train HMM models
trained_models = train_models(train_audio_files, AUDIOSET_FOLDER, num_words, num_states, feature_dimension, aii, aii_plus_1, num_iterations);




% Save trained models
save(fullfile(OUT_FOLDER, 'trained_hmm_models.mat'), 'trained_models');