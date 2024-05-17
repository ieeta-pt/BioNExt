# BioNExt

This repository contains the implementation for the work **Towards Discovery: An End-to-End System for Uncovering Novel Biomedical Relations**

# Table of Contents
1. [How to Use](#how-to-use)
   - [Install Dependencies](#install-dependencies)
   - [Run the System (Inference)](#run-the-system-inference)
   - [Select Which Modules to Run](#select-which-modules-to-run)
   - [How to Use LLM for Sequence Variant Detection in the Linker](#how-to-use-llm-for-sequence-variant-detection-in-the-linker)
   - [Complete Arguments Description](#complete-arguments-description)
2. [How to train](#how-to-train)
   - [Tagger Model](#tagger-model)
   - [Extractor Model](#extractor-model)
3. [Models](#models)
4. [Authors](#authors)

## How to use

### Install dependencies

The source code runs on Python. All dependencies are listed in `requirements.txt`. Data and models are automatically downloaded upon the first execution of the pipeline.

To install the dependencies, consider creating a virtual environment:

```bash
python -m venv my-virtual-venv
source my-virtual-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the System (inference)


The main entry point for our system is the `main.py` script. This script enables you to run the entire pipeline on your documents or datasets. Additionally, you can choose to execute specific modules, such as tagging or extraction, as needed.

For example, if you wish to process the BC8 BioRED track test set, simply provide the respective test set in BioC JSON format:
```bash
python main.py dataset/bc8_biored_task2_test.json
```
Please note that the bc8_biored_task2_test.json file is not included in this repository. However, it can be easily obtained from the following link: [BC8 BioRED Subtask 2 Test Set](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC8-BioRED-track/BC8_BioRED_Subtask2_Test_Set.zip).

Alternatively, there is a simple API for running our pipeline on any article currently indexed on PubMed. Simply use the keyword PMID: followed by the article identifier:

```bash
python main.py PMID:36516090
```

### Select which modules to run

You can specify which modules to run using the following flags:

* `-t` or `--tagger` enables the usage of the tagger module
* `-l` or `--linker` enables the usage of the linker module
* `-e` or `--extractor` enables the usage of the extractor module

By default, all modules are run if no flags are specified.


### How to Use LLM for Sequence Variant Detection in the Linker

Using Large Language Models (LLMs) for sequence variant detection can be complex, so we've made this component optional to accommodate all users. If you have access to an LLM, you can leverage it by extending the `GenericAPICall` class and implementing the required logic in the `run` method, which sends a prompt to the LLM. The `OllamaAPICall.py` is an example of how to use our LLMs powered by OLLAMA.

The integration of LLMs for sequence variant detection is entirely optional. Whether or not you configure the LLM component, sequence variant detection will always occur during linking by using direct matching techniques. This ensures that all users can benefit from essential functionality, even without LLM setup.

#### Setup and Configuration
To utilize LLMs in the pipeline:
1. **Extend the GenericAPICall Class**: Customize the `GenericAPICall` class by implementing the `run` method to handle the specifics of your LLM. For example, the `OllamaAPICall` class is designed to send the given prompts to an OLLAMA server, handling details like API endpoints, temperature settings, and GPU allocations.
   
2. **Specify the LLM Details in Command Line**: When running the main pipeline, specify the LLM API details using command-line arguments. For instance:
   ```bash
   python main.py PMID:36516090 --linker.llm_api.address http://IP:PORT --linker.llm_api.module OllamaAPICall
   ```
### Complete Arguments Description

#### Global Settings
These settings relate to the general configuration of the pipeline:
- `source_file`: Path to the input file, either a BIOC JSON or a PubMed ID prefixed with `PMID:`.
- `-t`, `--tagger`: Enable the tagger module.
- `-l`, `--linker`: Enable the linker module.
- `-e`, `--extractor`: Enable the extractor module.

#### Tagger Settings
Settings specific to the tagger module:
- `--tagger.checkpoint`: Path or identifier for the tagger model checkpoint (default: "IEETA/BioNExt-Tagger").
- `--tagger.trained_model_path`: Directory to where the tagger will be downloaded (default: "trained_models/tagger").
- `--tagger.batch_size`: Batch size for processing (default: 8).
- `--tagger.output_folder`: Directory for saving output from the tagger (default: "outputs/tagger").

#### Linker Settings
Settings specific to the linker module:
- `--linker.llm_api.module`: Module for the LLM API (default: None).
- `--linker.llm_api.address`: Address for the LLM API (default: None).
- `--linker.kb_folder`: Directory to where the knowledge bases will be downloaded (default: "knowledge-bases/").
- `--linker.dataset_folder`: Directory to where the datasets will be downloaded (default: "dataset/").
- `--linker.output_folder`: Directory for saving output from the linker (default: "outputs/linker").

#### Extractor Settings
Settings specific to the extractor module:
- `--extractor.output_folder`: Directory for saving output from the extractor (default: "outputs/extractor").
- `--extractor.checkpoint`: Path or identifier for the extractor model checkpoint (default: "IEETA/BioNExt-Extractor").
- `--extractor.trained_model_path`: Directory to where the tagger will be downloaded (default: "trained_models/extractor").
- `--extractor.batch_size`: Batch size for processing (default: 128).

These options allow you to customize the execution of the pipeline to suit your specific needs, whether running the full suite of tools or individual components.

## How to Train

### Tagger Model

The training environment for the Tagger model is set up under the `src/tagger` directory. To begin training, navigate to this directory and use the `hf_training.py` script as the main entry point. This script is built upon the Hugging Face `Trainer` API, allowing for straightforward model training with BERT architectures.

#### Training Command
To start the training process, use the following command:

```bash
cd src/tagger
python hf_training.py michiyasunaga/BioLinkBERT-base --augmentation unk --context 64
```

#### Parameters Description
The `hf_training.py` script allows for several arguments to customize the training process:

- **Model Checkpoint**: The first parameter specifies the pre-trained BERT model checkpoint to use as a starting point. In this example, `michiyasunaga/BioLinkBERT-base` is used.
- `--augmentation`: The type of data augmentation to apply (default: None). (unk and random are the options)
- `--p_augmentation`: Probability of applying augmentation on a per-example basis (default: 0.5).
- `--percentage_tags`: The percentage of tags (>0) to be augmented per sample (default: 0.2).
- `--context`: Length of the context window (default: 64 tokens).
- `--epochs`: Number of training epochs (default: 30).
- `--batch`: Batch size for training (default: 8).
- `--random_seed`: Random seed for reproducibility (default: 42).

#### Dataset
By default we are using the datasets under the `dataset` folder. In case that its empty consider running our system in inference mode (see above), since it will automaticly download the BioRED dataset used for this work.

### Extractor Model

The training setup for the Extractor model is similarly straightforward and utilizes the `hf_training.py` script under the `src/extractor` directory. This model is also trained using the Hugging Face `Trainer` API, tailored to handle the specific needs of relation extraction tasks in biomedical texts.

#### Training Command
To start training the Extractor model, navigate to the extractor directory and execute the following command:

```bash
cd src/extractor
python hf_training.py michiyasunaga/BioLinkBERT-base --novel
```

#### Parameters Description
The `hf_training.py` script for the Extractor model allows several arguments for customizing the training process:

- **Model Checkpoint**: Specifies the pre-trained BERT model checkpoint to be used. In this case, `michiyasunaga/BioLinkBERT-base` is the chosen model.
- `--arch`: Defines the architecture type for the model, with the default set to "mha" (multi-head attention). (bilstm is the other alternative)
- `--index_type`: Specifies the type of indexing used in the model, defaulting to "both". ("s" and "e" are the other alternatives)
- `--name`: Provides a unique name for identifying the training session or model version, default is None.
- `--epochs`: Determines the number of training epochs, set to 30 by default.
- `--batch`: Configures the batch size for training, default is 10.
- `--random_seed`: Sets a random seed to ensure reproducibility, default is 42.
- `--novel`: Enables joint training for relation and novelty detection. Activating this flag integrates novelty detection into the training process, enhancing the model's ability to distinguish between known and novel relations in the dataset.

#### Dataset
Ensure that the `dataset` folder contains the appropriate data before starting training. If the folder is empty, consider running the system in inference mode, which will automatically download the necessary BioRED dataset used for this work.

## Models

Our tagger and extractor models are integrated with the Hugging Face library and can be accessed and used in isolation at:

* tagger: https://huggingface.co/IEETA/BioNExt-Tagger
* extractor: https://huggingface.co/IEETA/BioNExt-Extractor

## **Authors:**

- Tiago Almeida ([ORCID: 0000-0002-4258-3350](https://orcid.org/0000-0002-4258-3350))
- Richard A A Jonker ([ORCID: 0000-0002-3806-6940](https://orcid.org/0000-0002-3806-6940))
- Rui Antunes ([ORCID: 0000-0003-3533-8872](https://orcid.org/0000-0003-3533-8872))
- João R Almeida ([ORCID: 0000-0003-0729-2264](https://orcid.org/0000-0003-0729-2264))
- Sérgio Matos ([ORCID: 0000-0003-1941-3983](https://orcid.org/0000-0003-1941-3983))
