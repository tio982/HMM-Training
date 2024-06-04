
function mfccs = extract_mfcc(audioFilePath, feature_dimension)
    % Check if the audio file exists before attempting to read it
    if ~isfile(audioFilePath)
        error('The audio file %s does not exist.', audioFilePath);
    end

    % Read the audio file
    [audio_in, fs] = audioread(audioFilePath);

    % Check if the audio is empty or not read properly
    if isempty(audio_in)
        error('The audio in file %s could not be read or is empty.', audioFilePath);
    end

    % Define parameters for MFCC extraction
    frameSize = round(30e-3 * fs); % 30ms frame size
    hopSize = round(10e-3 * fs);   % 10ms hop size

    % Check for valid frame and hop sizes
    if frameSize <= 0 || hopSize <= 0
        error('Frame size and hop size must be positive integers.');
    end

    % Create a window function (e.g., Hamming window)
    window = hamming(frameSize);

    % Check if the audio length is sufficient for MFCC processing
    if length(audio_in) < frameSize
        error('The audio file %s is too short for the given frame size.', audioFilePath);
    end

    % Extract MFCCs without delta features
    try
        mfccs = mfcc(audio_in, fs, ...
            'Window', window, ...
            'OverlapLength', frameSize - hopSize, ...
            'NumCoeffs', feature_dimension, ...
            'LogEnergy', 'Ignore');
    catch ME
        error('MFCC extraction failed for file %s: %s', audioFilePath, ME.message);
    end
end