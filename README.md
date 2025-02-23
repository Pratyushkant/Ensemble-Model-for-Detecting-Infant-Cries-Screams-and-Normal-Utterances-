# Ensemble Model for Detecting Infant Cries, Screams and Normal Utterances

## Audio Data Preprocessing Pipeline

### Overview
This Jupyter notebook contains scripts for preprocessing audio datasets, including counting and filtering files, segmenting and enhancing audio, and splitting the processed data into training, validation, and test sets.

### Dataset Structure
The datasets are stored in the following directories:
- `Dataset/Cry_Datasets/` - Contains crying audio samples
- `Dataset/Screaming_Datasets/` - Contains screaming audio samples
- `Dataset/Normal_Utterances_Datasets/` - Contains normal speech samples

### Scripts
#### 1. Data Preprocessing
##### `count_files_and_find_same_names()`
- Counts the number of valid audio files (`.wav`, `.mp3`) in each dataset.
- Identifies duplicate filenames across datasets.
- Computes the percentage of non-audio files removed.

##### `preprocess_audio(file_path, target_sr=16000)`
- Loads audio files in mono with a sampling rate of 16 kHz.

##### `segment_audio(audio, segment_length=2, sr=16000)`
- Segment audio into 2-second chunks.

##### `advanced_preprocess(segment, sr=16000)`
- Applies Wiener filtering for noise reduction.
- Applies Gaussian smoothing.
- Normalizes audio amplitude.

##### `process_dataset_in_batches(dataset_path, label, sr=16000, batch_size=50)`
- Processes audio files in batches to avoid memory overload.
- Segments, enhances and saves audio.
- Stores metadata in `processed_audio_metadata.csv`.

#### 2. Data Balancing
- Ensures each class has an equal number of samples by downsampling.
- Computes class weights for model training.

#### 3. Data Splitting
##### `train_test_split`
- Splits data into 70% training, 15% validation, and 15% test while maintaining class balance.
- Saves metadata for each split.

##### `copy_files(split_df, destination)`
- Copies processed audio files to respective split directories (`Split_Data/train/`, `Split_Data/val/`, `Split_Data/test/`).

### Outputs
- `Processed_Audio/` - Stores processed audio segments.
- `processed_audio_metadata.csv` - Metadata of processed files.
- `Split_Data/` - Stores train, validation, and test datasets.
- `train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv` - Metadata files for each split.

### Notes
- Ensure that all dataset directories exist before running the scripts.
- The preprocessing may take time depending on dataset size.

## Training Wav2Vec2 Model

### Overview
This Jupyter Notebook is for training a Wav2Vec2 model for automatic speech recognition (ASR) or related tasks. The notebook guides users through loading datasets, preprocessing audio data, fine-tuning the model, and evaluating its performance.

### Notebook Structure
The notebook is divided into the following sections:

1. **Setup & Imports**: Loads necessary libraries and sets up the environment.
2. **Dataset Preparation**: Loads and preprocesses the audio dataset.
3. **Feature Extraction**: Uses Wav2Vec2 feature extractor for speech-to-text conversion.
4. **Model Fine-Tuning**: Trains the Wav2Vec2 model on the dataset.
5. **Evaluation**: Tests the model's performance using word error rate (WER) and character error rate (CER).
6. **Inference**: Demonstrates real-world predictions on sample audio inputs.
7. **Saving & Exporting**: Saves the trained model for later use.

### Expected Output
- Trained Wav2Vec2 model
- Performance metrics (WER, CER)
- Transcribed text from audio samples

### Customization
- Replace the dataset with your own audio data.
- Adjust training parameters such as learning rate, batch size, and epochs.
- Implement different evaluation metrics if needed.

### Troubleshooting
- Ensure you have enough GPU memory for training.
- Check for dataset formatting errors.
- Adjust learning rates if training is unstable.

### References
- [Hugging Face Wav2Vec2 Documentation](https://huggingface.co/transformers/model_doc/wav2vec2.html)
- [Speech Datasets on Hugging Face](https://huggingface.co/datasets)
