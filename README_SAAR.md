
# Feature Extraction Script

## Description
- **Inputs**:
  - `audio_files`: A list or array of audio file names.
  - `AUDIOSET_FOLDER`: The directory where the audio files are stored.
  - `OUT_FOLDER`: The output directory to save the extracted features.
  - `OUT_SUBFOLDER`: A subfolder within `OUT_FOLDER` for organization.

## Parallel Pool initialisation
- Checks if MATLAB’s parallel computing toolbox is available.
- Initialises a parallel pool if it's not already running.
- Enables parallel execution of the feature extraction process.

## Preallocation for Output
- Preallocates a cell array `extracted_features`.
- Stores the MFCC features for each audio file.
- The length of `extracted_features` equals the number of audio files.

## Parallel Processing of Audio Files
- Uses a `parfor` loop to process each audio file.
- Constructs the full path of each audio file using `fullfile`.
- Extracts the MFCC features using `extract_mfcc`.
- Stores the features in the preallocated `extracted_features` array.

## Output and Saving
- Iterates over the audio files after processing.
- Constructs a new file name for each, replacing the original extension with `.mat`.
- Saves the MFCC features in a `.mat` file in the specified output directory.

## Dependencies and Requirements
- Requires MATLAB’s Parallel Computing Toolbox.
- Calls `extract_mfcc` for feature extraction.



# Extract MFCC Script

## Description
- **Purpose**: Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.
- **Input**: `audioFilePath` - The path to the audio file.

## Read the Audio File
- Uses `audioread` to load the audio file.
- Stores the audio signal in `audio_in` and the sampling frequency in `fs`.

## Define Parameters for MFCC Extraction
- `frameSize`: Sets the frame size to 30 milliseconds (ms) of the audio signal.
- `hopSize`: Sets the hop size to 10 ms.
- `numCoeffs`: Specifies the number of cepstral coefficients to extract, set to 13.

## Create a Window Function
- Applies a Hamming window to each frame with `window = hamming(frameSize)`.

## Extract MFCCs
- Calls the `mfcc` function with the audio signal and sampling frequency.
- Passes additional parameters:
  - `Window`: The Hamming window.
  - `OverlapLength`: The overlap between frames.
  - `NumCoeffs`: The number of coefficients.
  - `LogEnergy`: Set to 'Ignore'.
- The result, `mfccs`, contains the MFCCs of the audio signal without delta features.

## Output
- `mfccs`: An array containing the extracted MFCCs for the audio file.



# Initialises HMM Models Script

## Purpose
- Initialises Hidden Markov Model (HMM) parameters for a specified number of words. Each model includes matrices for state transitions (A), emission probabilities (B), initial state distribution (pi), and parameters for emission distributions (means and variances).

## Inputs
- `num_words`: Number of different words to model.
- `num_states`: Total number of states in each HMM.
- `num_occupational_states`: Number of states that emit observations.
- `feature_dimension`: Dimension of the feature space for emissions.
- `aii`: Probability of staying in the same state.
- `aii_plus_1`: Probability of transitioning to the next state.

## Process

1. **Initialises Structure Array**
   - Creates a structure array `models` with fields `A`, `B`, `means`, `variances`, and `pi`.

2. **Initialises Each Model**
   - For each word (model):
     - Initialises the transition matrix `A` using `initialise_a_matrix`.
     - Initialises the emission probability matrix `B` using `initialise_b_matrix`.
     - Initialises the initial state distribution `pi` as a uniform distribution.
     - Initialises `means` and `variances` for emission probabilities using `initialise_mean_and_variance`.

3. **Initialise_a_matrix Function**
   - Constructs a transition probability matrix with self-loop probabilities and probabilities for transitioning to the next state.
   - The last state is designated as an exit state.

4. **Initialise_b_matrix Function**
   - Constructs an emission probability matrix with uniform distributions over features for emitting states and a uniform value for non-emitting states.

5. **Initialise_pi_vector Function**
   - Creates a uniform initial state distribution vector.

6. **Initialise_mean_and_variance Function**
   - Randomly Initialises the means and variances for emission probabilities, ensuring non-zero variances.

## Output
- `models`: A structure array containing initialised parameters for each word's HMM.


## Implementation Notes
- The uniform initialisation of `pi` reflects an assumption of no prior knowledge about the initial state.
- The initialisation of `A` and `B` respects the probabilistic constraints of HMMs, ensuring that each row of these matrices sums to one.
- The means and variances for emission distributions are initialised randomly, providing a starting point for further refinement during model training.


# Backward Algorithm Script

## Purpose
- Implements the backward algorithm for Hidden Markov Models (HMMs).
- Computes the backward probabilities (`beta`) for each state at each time step in the observed sequence, incorporating a scaling factor to maintain numerical stability.

## Inputs
- `matrix_a`: State transition probability matrix.
- `matrix_b`: Observation emission probability matrix.
- `observed_sequence`: Sequence of observed symbols (indices).
- `scaling_factor`: Array of scaling factors obtained from the forward algorithm.

## Process

1. **initialisation**
   - `num_states`: Number of states in the HMM, determined from the size of `matrix_a`.
   - `sequence_length`: Length of the observed sequence.
   - Initialises the `beta` matrix (backward probabilities) and sets the last column according to the scaling factor.

2. **Backward Iteration**
   - Iterates backward through the sequence from `sequence_length-1` to `1`.
   - For each state at time `t`, calculates the sum of the product of transition probabilities, emission probabilities for the next observed symbol, and backward probabilities at time `t+1`.
   - Scales the `beta` values for each time step using the corresponding scaling factor.

## Output
- `beta`: Matrix containing the scaled backward probabilities for each state at each time step.


## Implementation Notes
- The scaling factor, typically derived from the forward algorithm, is critical for preventing numerical underflow, a common issue in probability calculations for long sequences.
- The computation of `beta` values with scaling factors ensures consistency with the scaled forward probabilities (`alpha`), maintaining the accuracy of subsequent calculations in HMM processes.


# Forward Algorithm Script

## Purpose
- Implements the forward algorithm for Hidden Markov Models (HMMs).
- Computes the forward probabilities (`alpha`) for each state at each time step in the observed sequence, along with a scaling factor to prevent numerical underflow.

## Inputs
- `matrix_a`: State transition probability matrix.
- `matrix_b`: Observation emission probability matrix.
- `observed_sequence`: Sequence of observed symbols (indices).

## Process

1. **initialisation**
   - `num_states`: Number of states in the HMM, determined from the size of `matrix_a`.
   - `sequence_length`: Length of the observed sequence.
   - Initialises the `alpha` matrix (forward probabilities) and `scaling_factor` array.

2. **First Time Step initialisation**
   - Computes initial `alpha` values for the first time step using the initial transition probabilities from `matrix_a` and the emission probabilities for the first observed symbol from `matrix_b`.
   - Calculates the initial scaling factor to normalise `alpha` values, preventing numerical underflow.

3. **Induction Step**
   - Iteratively calculates `alpha` for each subsequent time step.
   - For each state at time `t`, computes the dot product of the previous `alpha` values and the corresponding transition probabilities, then multiplies by the emission probability of the observed symbol at time `t`.
   - Calculates and applies the scaling factor for each time step to normalise `alpha` values.

## Output
- `alpha`: Matrix containing scaled forward probabilities for each state at each time step.
- `scaling_factor`: Array of scaling factors used at each time step.


## Implementation Notes
- The scaling factor helps to prevent underflow, a common issue when dealing with probabilities in sequences of substantial length.


# Baum-Welch Training Algorithm Script

## Purpose
- Implements the Baum-Welch algorithm for training Hidden Markov Models (HMMs).
- Adjusts the model parameters to maximize the probability of the observed sequences.

## Inputs
- `observed_sequences`: Observed sequences, either a vector or a matrix.
- `initial_A`: Initial state transition probability matrix.
- `initial_B`: Initial observation emission probability matrix.
- `num_iterations`: Number of iterations for the algorithm.

## Process

1. **Validation**
   - Checks if `initial_A` is a square matrix and if `initial_B` has the same number of rows as states in `initial_A`.
   - Validates that `observed_sequences` is either a vector or a matrix.

2. **initialisation**
   - Initialises `A` and `B` matrices with `initial_A` and `initial_B`.
   - Prepares arrays to store `gamma` and `xi` values for each sequence.

3. **Baum-Welch Iterations**
   - For each iteration, processes each observed sequence:
     - Runs forward (`alpha`) and backward (`beta`) algorithms with scaling.
     - Calculates `gamma` (state occupancy probabilities) and `xi` (state transition probabilities).
     - Updates sums for `A` and `B` matrices based on `gamma` and `xi`.
   - Normalises the accumulated sums to update `A` and `B` matrices.

4. **Handling Different Sequence Formats**
   - Adjusts processing based on whether `observed_sequences` is a vector or a matrix.

5. **Gamma and Xi Calculations**
   - Computes `gamma` and `xi` for each time step in each sequence.
   - Stores `gamma` and `xi` for each sequence.

6. **Matrix Updates**
   - Updates the `A` matrix by normalising the accumulated transition probabilities.
   - Updates the `B` matrix by normalising the accumulated emission probabilities.

## Output
- `updated_A`: Updated state transition probability matrix.
- `updated_B`: Updated observation emission probability matrix.
- `all_gamma`: Collection of `gamma` values for each sequence.
- `all_xi`: Collection of `xi` values for each sequence.


## Implementation Notes
- The function can handle input sequences in different formats, adding flexibility.
- It includes handling for zero denominator cases in `gamma` and `xi` calculations, ensuring numerical stability.
- The normalisation of `A` and `B` ensures that the stochastic nature of the HMM is maintained.


## Implementation Notes
- The normalisation of `A` and `B` is used to maintain the probabilistic interpretation of the matrices.


# HMM Training Script

## Purpose
- Trains Hidden Markov Models (HMMs) for a set of audio files, each representing a different word.

## Inputs
- `audio_files`: A list or array of audio file names.
- `AUDIOSET_FOLDER`: The directory where the audio files are stored.
- `OUT_FOLDER`: The directory where extracted features are to be saved.
- `OUT_SUBFOLDER`: A subfolder for organisation within `OUT_FOLDER`.
- `num_words`: The number of words (and thus the number of HMMs) to train.
- `num_states`: The total number of states for each HMM.
- `num_emitting_states`: The number of emitting states for each HMM.
- `feature_dimension`: The dimensionality of the feature vectors.
- `aii`: The probability of staying in the same state.
- `aii_plus_1`: The probability of transitioning to the next state.
- `num_iterations`: The number of iterations for the Baum-Welch algorithm.

## Process

1. **Load Cluster Centers**
   - Loads the `cluster_centers.mat` file containing `clusterCenters`.

2. **Extract MFCC Features**
   - Calls `feature_extraction` to process `audio_files` and extract MFCC features.

3. **Initialise HMM Models**
   - Initialises HMM models with the given parameters using `initialise_hmm_models`.

4. **Model Training**
   - Iterates over each word to train its corresponding HMM.
   - Validates that the MFCC features for each word are in matrix form.
   - Quantises MFCC features to a sequence of cluster indices using `map_mfcc_to_clusters`.
   - Trains each model's parameters (`A` and `B`) using the `baum_welch_training` function.
   - Updates the model parameters with the trained values.

## Output
- `trained_models`: The array of trained HMM models with updated parameters.


# MFCC Feature Extraction and Clustering Script

## Purpose
- Extracts Mel-frequency cepstral coefficients (MFCCs) from a collection of audio files and performs k-means clustering to create a codebook of cluster centers.

## Process

1. **Retrieve Audio Files**
   - Gathers a list of all `.mp3` audio files in the `AUDIOSET_FOLDER`.
   - Stores the filenames in `audio_files`.

2. **Initialise Feature Storage**
   - Sets up an empty matrix `all_mfcc_features` to hold the MFCC features from all files.

3. **Feature Extraction Loop**
   - Loops over each audio file.
   - Constructs the full path to the file.
   - Extracts MFCC features using the `extract_mfcc` function.
   - Accumulates the extracted features into `all_mfcc_features`.

4. **K-Means Clustering**
   - Ensures `all_mfcc_features` is formatted correctly for k-means (rows as feature vectors).
   - Applies k-means clustering to group the MFCC features into `num_clusters` clusters.
   - Stores the cluster centers in `clusterCenters`.

5. **Save Cluster Centers**
   - Saves the computed `clusterCenters` to a `.mat` file for later use.

## Output
- A file `cluster_centers.mat` containing the cluster centers that represent the codebook for the set of MFCC features.


## Note
- `num_clusters` should be defined in the context where this script is used, indicating the desired number of clusters for the k-means algorithm.


# Complete Project Script for HMM Training

## Purpose
- Implements a complete workflow for training Hidden Markov Models (HMMs) using MFCC features extracted from a set of audio files. The process includes data preparation, feature extraction, model initialisation, training, and saving the trained models.

## Process

1. **Folder Paths and File Retrieval**
   - Sets paths for the dataset, output folder, and subfolder.
   - Retrieves the names of all `.mp3` audio files from the dataset folder.

2. **Data Splitting**
   - Splits the list of audio files into training and testing sets based on a 90:10 ratio.

3. **Load Cluster Centers**
   - Loads precomputed `cluster_centers.mat` containing `clusterCenters`.

4. **Define HMM Parameters**
   - Sets parameters for the HMM models, including the number of words, states, emitting states, and feature dimension. Also sets transition probabilities and the number of training iterations.

5. **Extract MFCC Features for Training Set**
   - Extracts MFCC features from the training audio files.

6. **Map MFCC Features to Clusters**
   - Maps each set of MFCC features to corresponding cluster indices to create observed sequences for training.

7. **Initialises HMM Models**
   - Initialises HMM models with specified parameters.

8. **Train HMM Models**
   - Iterates over each word (each HMM model).
   - Validates that the observed sequences are in the correct format.
   - Trains each model using the Baum-Welch algorithm.
   - Updates the model parameters (`A`, `B`) with the trained values.

9. **Save Trained Models**
   - Saves the trained HMM models to a file for future use.

10. **Testing and Evaluation**
   - Outlines further steps for testing and evaluating the trained models, which may include extracting and mapping test MFCC features and evaluating the models' performance.

## Output
- `trained_hmm_models.mat`: A file containing the trained HMM models.



## Implementation Notes
- The script includes crucial steps of validation to ensure data integrity.
- Feature extraction and quantisation are essential for converting continuous audio signals into a format suitable for HMM training.
- The Baum-Welch algorithm is the core of the training process, requiring careful implementation and parameter tuning.



