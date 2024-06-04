
function extracted_features = feature_extraction(audio_files, AUDIOSET_FOLDER, feature_dimension)
    % Start parallel pool outside this function if running feature_extraction multiple times
    % Ensure Distributed Computing Toolbox is available for parallel processing

    % Extract MFCC features
    mfcc_features = cell(1, numel(audio_files));
    for filenum = 1:length(audio_files)
        mfcc_features{filenum} = process_file(filenum, audio_files, AUDIOSET_FOLDER, feature_dimension);
        
        % Optional: Check dimensions right after extraction (can be removed if you're confident about the extract_mfcc function)
        if size(mfcc_features{filenum}, 2) ~= feature_dimension
            warning('MFCC extraction returned unexpected number of coefficients for file %d.', filenum);
        end
    end

    % Return the extracted features
    extracted_features = mfcc_features;
end

function mfcc_features = process_file(filenum, audio_files, AUDIOSET_FOLDER, feature_dimension)
    current_fname = audio_files{filenum};
    audiofname_full = fullfile(AUDIOSET_FOLDER, current_fname);
    
    % Log the processing status
    disp(['Processing file ' num2str(filenum) ' of ' num2str(numel(audio_files)) ': ' current_fname]);

    % Extract MFCC features and handle any errors
    try
        mfcc_features = extract_mfcc(audiofname_full, feature_dimension);
        if isempty(mfcc_features)
            mfcc_features = zeros(0, feature_dimension); % Placeholder for maintaining dimension consistency
        end
    catch ME
        disp(['Failed to process file ' current_fname ' due to error: ' getReport(ME)]);
        mfcc_features = zeros(0, feature_dimension); % Placeholder for maintaining dimension consistency
    end
end