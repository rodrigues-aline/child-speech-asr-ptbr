# child-speech-asr-ptbr
Fine-tuning Wav2Vec 2.0 and XLS-R for low-resource Automatic Speech Recognition on Brazilian Portuguese children's speech. Includes code, models, and interpretability analysis.


This repository provides the necessary resources for reproducing the experiments and continuing this research. It includes links to the pre-trained and fine-tuned models, language models, source codes for training, inference, interpretability analysis, and the dataset used.

## Language Models (LMs)

The n-gram Language Models (LMs) used to enhance ASR model decoding were created with the **KenLM Language Model Toolkit**. The LMs were trained with the combined data from the training and validation sets of the child speech corpus, separated by type (words for narrative texts and phonemes for isolated words).

The `.arpa` files and the code used for building these LMs are available in the following GitHub repository:
link:

## Grapheme to Phoneme (G2P)

The grapheme-to-phoneme conversion of the isolated word corpus was performed using the following repository: [falabrasil](https://github.com/falabrasil/annotator)



## ASR Models (Automatic Speech Recognition)

The pre-trained and fine-tuned ASR models are based on the **Wav2Vec2.0** architecture and are hosted on the **HuggingFace Hub**, allowing easy access and use through the **transformers** library.

### Pre-trained Models (Baseline)

The models used in the baseline phase, fine-tuned on adult Brazilian Portuguese speech data, can be accessed at the following links:

* **CORAA:** [wav2vec2-large-xlsr-coraa-portuguese](https://huggingface.co/Edresson/wav2vec2-large-xlsr-coraa-portuguese)
* **grosman-53:** 
  [wav2vec2-large-xlsr-53-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese)
* **grosman-1b:** [wav2vec2-xls-r-1b-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-portuguese)
* **facebook-53 (Multilingual):** [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
* **facebook-300m (Multilingual):** [wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* **facebook-1b (Multilingual):** [wav2vec2-xls-r-1b](https://huggingface.co/facebook/wav2vec2-xls-r-1b)


### Fine-tuned Models

The models that achieved the best performance after fine-tuning for each specific task are listed below. All fine-tuned models are available on the HuggingFace Hub.

#### Best Model for Narrative Texts

The model that achieved the lowest Word Error Rate (WER) of 0.0836 was **`grosman-1b-aug`**, a fine-tuned version of **`jonatasgrosman/wav2vec2-xls-r-1b-portuguese`** using the child speech corpus and data augmentation strategy.
Link: [wav2vec2-large-xlsr-grosman-1b-aug-texts-exp-1](https://huggingface.co/alinerodrigues/wav2vec2-large-xlsr-grosman-1b-aug-texts-exp-1)

#### Best Model for Isolated Words (Phonemic Approach)

For the phonemic transcription task, the model that obtained the lowest Phoneme Error Rate (PER) of 0.0437 was **`CORAA-aug`**, fine-tuned from **`Edresson/wav2vec2-large-xlsr-coraa-portuguese`** with the child speech corpus and data augmentation strategy.
Link: [wav2vec2-large-xlsr-coraa-aug-words-phoneme-exp-1-v6](https://huggingface.co/alinerodrigues/wav2vec2-large-xlsr-coraa-aug-words-phoneme-exp-1-v6)


## Child Speech Corpus

The child speech corpus developed for this dissertation is a valuable resource for future research. Due to the sensitive nature of children’s data, access is restricted and requires formal authorization.

* **Corpus Access Request:**
  To request access to the corpus, please contact [Aline N. Rodrigues / Carlos H. C. Ribeiro] via email at [rodrigues.aline.n@gmail.com](mailto:.aline.n@gmail.com) / [carlos51@gmail.com](mailto:carlos51@gmail.com).
  Access will be granted upon signing a Data Use and Confidentiality Agreement.


## Source Code

The source code developed for this research scripts for data preprocessing, model training, inference, and interpretability analysis—is available in this main GitHub repository.

The repository is organized as follows:

* **`data_preprocessing/`**: Scripts for audio segmentation, data augmentation, and dataset splitting (train/validation/test).
* **`lm_training/`**: Scripts for building n-gram Language Models with KenLM.
* **`model_finetuning/`**: Code for fine-tuning Wav2Vec2.0 models, including weight reinitialization and phased training strategies.
* **`inference/`**: Scripts for ASR inference, decoding hyperparameter optimization ($\alpha$ and $\beta$), and computation of WER, CER, and PER metrics.

The implementation of embedding extraction, dimensionality reduction (UMAP, t-SNE, PCA), and clustering (K-Means, Spherical K-Means) techniques for model interpretability analysis is available here: [wav2vec2_interpretation](https://github.com/rodrigues-aline/wav2vec2_interpretation)


## Environment Requirements

### Hardware Requirements

Hardware requirements according to the processing stage:

* **Training and Fine-tuning:**
  As detailed in the methodology, all training experiments were conducted in a cloud computing environment (**Google Colaboratory**) equipped with an **NVIDIA A100 GPU** (40 GB VRAM). This high-performance hardware is essential for fine-tuning the large-scale models described in this dissertation.

* **Inference:**
  For efficient batch inference, GPU usage is strongly recommended. However, individual sample inference can be performed on a modern CPU, albeit with higher latency.

* **Preprocessing:**
  The remaining scripts (preprocessing, interpretability) have no specific hardware requirements and can be executed on a standard CPU.

### Software Requirements

The software environment is based on **Python (version 3.8 or higher)**.
To ensure full reproducibility of the experiments, all libraries and their exact versions are listed in the `requirements.txt` file.
