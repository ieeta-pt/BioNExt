# BioNExt

This repository contains the implementation for the work as described in:

**Towards Discovery: An End-to-End System for Uncovering Novel Biomedical Relations**

**Authors:**

- Tiago Almeida ([ORCID: 0000-0002-4258-3350](https://orcid.org/0000-0002-4258-3350))
- Richard A A Jonker ([ORCID: 0000-0002-3806-6940](https://orcid.org/0000-0002-3806-6940))
- Rui Antunes ([ORCID: 0000-0003-3533-8872](https://orcid.org/0000-0003-3533-8872))
- João R Almeida ([ORCID: 0000-0003-0729-2264](https://orcid.org/0000-0003-0729-2264))
- Sérgio Matos ([ORCID: 0000-0003-1941-3983](https://orcid.org/0000-0003-1941-3983))


## Abstract
Biomedical relation extraction is an ongoing challenge within the Natural Language Processing (NLP) community. Its application is important for understanding scientific biomedical literature, with many use cases, such as drug discovery, precision medicine, disease diagnosis, treatment optimization, and biomedical knowledge graph construction. Therefore, the development of a tool capable of effectively addressing this task holds the potential to improve knowledge discovery by automating the extraction of relations from research manuscripts. The first track in the BioCreative VIII competition extended the scope of this challenge by introducing the detection of novel relations within literature. This paper describes our participation system initially focused on jointly extracting and classifying novel relations between biomedical entities. We then describe our subsequent advancement to an end-to-end model. Specifically, we enhanced our initial system by incorporating it into a cascading pipeline that includes a tagger and linker module. This integration enables the comprehensive extraction of relations and classification of their novelty directly from raw text. Our experiments yielded promising results, our tagger module managed to attain state-of-the-art NER performance, with a micro F1-score of 90.24, whilst our end-to-end system achieved a competitive novelty F1-score of 24.59. 

## Setup
The setup script installs the virtual environment, the model, and the embeddings. Please refer to the provided setup script for installation.

TODO

## Dataset
The dataset used to train the models can be found [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC8-BioRED-track/).

## Model
The model used as the tagger can be found [here](), and the extractor can be found [here]() .

## Embeddings
The embeddings created by SAPBERT can be found [here](https://zenodo.org/records/11126786). These embeddings were created using the SAPBERT multilingual large model ([SAPBERT-UMLS-2020AB-all-lang-from-XLMR-large](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large)).

## Usage
```python main.py PATH_TO_JSON_FILE --tagger.checkpoint  trained_models/tagger/TODO --linker.kb_folder ./embeddings --linker.llm_api.address IP --linker.llm_api.module OllamaAPICall --extractor.checkpoint trained_models/extractor/TODO
```

### Arguments Description:

<!-- trained_models/tagger/BioLinkBERT-large-dense-60-2-unk-P0.25-0.75-42-full/checkpoint-1200 -->
- `main.py`: The main script for executing the system.
- `PATH_TO_JSON_FILE`: Path to the BIOCJSON file.
- `--tagger.checkpoint`: Path to the checkpoint of the tagger module.
- `--linker.kb_folder`: Path to the folder containing the knowledge resources, please see the Embeddings section above.
- `--linker.llm_api.address`: Address for the LLM API.
- `--linker.llm_api.module`: Module for the LLM API.
- `--extractor.checkpoint`: Path to the checkpoint of the extractor module.


The code is designed to work in a modular fashion, hwere each aspect (tagger, linker, extractor) can be run in isolation. With regards to the linker
<!-- Please ensure all necessary dependencies are installed and paths are correctly configured before running the script. -->


