# CS60075 NLP Assignment II - Project 5 (Dataset 5.3: UD English LinES)

This repository contains the Google Colab notebook for **Project 5: From Static to Contextual - Distributional Semantics and What's Inside BERT**. The project is implemented using **Dataset 5.3 (UD English LinES)** and is designed to be executed end-to-end with minimal manual intervention.

## 1. Environment Setup

This code is optimized to run seamlessly in **Google Colab**. It requires a GPU runtime for efficient processing of Task 2 and Task 3.

**To set up the Colab environment:**
1. Open the provided `.ipynb` notebook in Google Colab.
2. Go to `Runtime` -> `Change runtime type` and ensure **Hardware accelerator** is set to **T4 GPU** (or any available GPU).
3. The notebook relies on the following core Python libraries. Colab pre-installs most of these, but they are explicitly managed within the notebook:
   - `torch`
   - `numpy`
   - `pandas`
   - `scipy`
   - `scikit-learn`
   - `transformers` (Hugging Face)
   - `datasets` (Hugging Face)

## 2. Dataset Preparation

The notebook is completely hands-free regarding dataset acquisition. It programmatically downloads and prepares all necessary data:

*   **Primary Dataset (UD English LinES):** Automatically downloaded and cached via the Hugging Face `datasets` library using `load_dataset("universal-dependencies/universal_dependencies", "en_lines")`.
*   **WordSim-353 (Shared Benchmark):** Automatically downloaded via `wget` from `http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip` and unzipped directly into the Colab environment.
*   **Google Analogy Set (Shared Benchmark):** Automatically downloaded via `wget` from `https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt`.

## 3. Pretrained Checkpoints Used

This project utilizes the following pretrained model for extracting contextual embeddings and running the structural probe:
*   **Model:** `bert-base-uncased`
*   **Source:** Hugging Face Model Hub (`transformers.AutoModel.from_pretrained("bert-base-uncased")`)

## 4. Execution Instructions

The notebook is divided into three distinct modules for Task 1, Task 2, and Task 3. It must be run sequentially, as Task 3 depends on the optimal layer ($l^*$) identified in Task 2.

**To run end-to-end:**
1. Click **Runtime -> Run all** in the Google Colab menu.
2. **Google Drive Authorization:** In the very first cell, you will be prompted to grant Google Drive access. This is strictly required for caching large matrix extractions and saving training checkpoints so that the pipeline does not have to restart from scratch if interrupted. Click "Connect to Google Drive" and allow permissions.
3. The notebook will then execute all three tasks automatically. Wait for execution to finish (Task 2 training will take the most time). 

## 5. Outputs and Checkpoints

All extracted features, training checkpoints, and generated matrices are saved automatically to the root of your Google Drive (`/content/drive/MyDrive/`). This enables auto-resuming.

The following files will be written to your Drive:
*   `train_vectors_12L.pt`: Cached 12-layer BERT embeddings for the training set.
*   `train_dists_12L.pt`: Cached gold tree distances for the training set.
*   `train_depths_12L.pt`: Cached gold tree depths for the training set.
*   `task2_probe_results.pt`: Incremental training checkpoint storing UUAS and Spearman metrics for the structural probes.
*   `bert_matrix_L8.npy`: The generated static matrix pooling the contextual vectors from the optimal BERT layer (Task 3). *(Note: "L8" may reflect a different number if the optimal layer changes).*

Results tables, performance plots, and qualitative examples will be printed directly to the standard output cells within the Colab notebook at the end of each task section.