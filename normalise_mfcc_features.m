
function normalised_features = normalise_mfcc_features(mfcc_feature_set, feature_dimension)
    % Check if mfcc_feature_set is a numeric array
    if ~isnumeric(mfcc_feature_set) || isempty(mfcc_feature_set)
        error('Input mfcc_feature_set must be a non-empty numeric array.');
    end
    
    % Check if the feature dimension is correct
    if size(mfcc_feature_set, 2) ~= feature_dimension
        error('Dimension mismatch in MFCC features. Expected %d, got %d.', feature_dimension, size(mfcc_feature_set, 2));
    end
    
    % Calculate the mean and standard deviation for normalization
    global_mean = mean(mfcc_feature_set, 1);
    global_std = std(mfcc_feature_set, 0, 1);
    
    % Replace zero standard deviations with 1 to avoid division by zero
    global_std(global_std == 0) = 1;
    
    % Perform normalization
    normalised_features = (mfcc_feature_set - global_mean) ./ global_std;
end