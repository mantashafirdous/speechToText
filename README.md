# Speech-to-Text Enabled Chatbot Using Transformer Model

## Overview

This project focuses on developing a Speech-to-Text (STT) Enabled Chatbot leveraging the Transformer architecture to provide real-time transcription and conversational capabilities. It addresses the limitations of traditional STT systems, such as handling long-range dependencies and contextual nuances, and introduces an AI-driven solution for diverse applications like healthcare, education, and assistive technologies.

## Features

* **Transformer-Based Model:** Utilizes OpenAI’s Whisper model for accurate transcription.

* **Self-Attention Mechanisms:** Captures long-range dependencies in speech data.

* **Robust Preprocessing:** Includes tokenization, noise reduction, and feature extraction.

* **Evaluation Metrics:** Implements Word Error Rate (WER) and Character Error Rate (CER) for performance validation.

* **Domain-Specific Fine-Tuning:** Optimized for medical transcription and other specialized use cases.

## Dataset

* **Name**: Medical Speech, Transcription, and Intent Dataset (Kaggle)

* **Description**:

* * Contains 6,661 audio clips with transcriptions and intent labels.

* * Includes diverse accents, medical terminologies, and noise conditions.

* **Format**: WAV audio files with associated metadata.

## Methodology

### Preprocessing:

* Metadata extraction and cleaning.

* Audio file validation and dataset splitting (train, test, validate).

* Conversion to Hugging Face Datasets for streamlined processing.

### Model Architecture:

* **Encoder**: Processes audio features with self-attention layers.

* **Decoder**: Generates text transcriptions using cross-attention mechanisms.

* **Output**: Token-by-token transcription through a softmax layer.

### Training:

* **Framework:** PyTorch.

* **Hardware:** NVIDIA Tesla T4 GPU on Google Colab.

* **Optimizer:** Adam with a learning rate of 1e-5.

* **Metrics:** WER and CER for transcription accuracy.

## Results

### Performance:

* **Achieved WER:** 21.33% on the Medical Speech dataset.

* Demonstrated robustness in noisy and multilingual settings.

### Comparative Analysis:

* Outperformed traditional models like RNNs and HMMs in transcription accuracy.

## Tools and Technologies

* **Frameworks:** PyTorch, Hugging Face Transformers.

* **Libraries:** Torchaudio for feature extraction.

* **Hardware:** NVIDIA Tesla T4 GPU, Google Colab.

* **Techniques:** Tokenization, positional encoding, noise reduction.

## Future Scope

* **Noise Robustness:** Incorporate advanced noise reduction techniques.

* **Multimodal Integration:** Combine visual and textual data for enhanced interactions.

* **Scalability:** Optimize for deployment on resource-constrained devices.

* **Dataset Diversity:** Include more languages, accents, and demographics.

