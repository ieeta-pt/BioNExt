# BioNExt

This repository contains the implementation for the work as described in:

**Towards Discovery: An End-to-End System for Uncovering Novel Biomedical Relations**

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

The main entry point for our system is the `main.py` script. This script allows you to run the entire pipeline on your documents or datasets. You can also choose to run specific modules, such as tagging or extraction, if needed.

To process a local dataset or document, which must be in BioC JSON format, use the following command. This command will run the complete pipeline on the specified dataset, utilizing all enabled modules (tagger, linker, and extractor by default).


```bash
python main.py dataset_in_bioc.json
```

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
## Complete Arguments Description

### Global Settings
These settings relate to the general configuration of the pipeline:
- `source_file`: Path to the input file, either a BIOC JSON or a PubMed ID prefixed with `PMID:`.
- `-t`, `--tagger`: Enable the tagger module.
- `-l`, `--linker`: Enable the linker module.
- `-e`, `--extractor`: Enable the extractor module.

### Tagger Settings
Settings specific to the tagger module:
- `--tagger.checkpoint`: Path or identifier for the tagger model checkpoint (default: "IEETA/BioNExt-Tagger").
- `--tagger.trained_model_path`: Directory to where the tagger will be downloaded (default: "trained_models/tagger").
- `--tagger.batch_size`: Batch size for processing (default: 8).
- `--tagger.output_folder`: Directory for saving output from the tagger (default: "outputs/tagger").

### Linker Settings
Settings specific to the linker module:
- `--linker.llm_api.module`: Module for the LLM API (default: None).
- `--linker.llm_api.address`: Address for the LLM API (default: None).
- `--linker.kb_folder`: Directory to where the knowledge bases will be downloaded (default: "knowledge-bases/").
- `--linker.dataset_folder`: Directory to where the datasets will be downloaded (default: "dataset/").
- `--linker.output_folder`: Directory for saving output from the linker (default: "outputs/linker").

### Extractor Settings
Settings specific to the extractor module:
- `--extractor.output_folder`: Directory for saving output from the extractor (default: "outputs/extractor").
- `--extractor.checkpoint`: Path or identifier for the extractor model checkpoint (default: "IEETA/BioNExt-Extractor").
- `--extractor.trained_model_path`: Directory to where the tagger will be downloaded (default: "trained_models/extractor").
- `--extractor.batch_size`: Batch size for processing (default: 128).

These options allow you to customize the execution of the pipeline to suit your specific needs, whether running the full suite of tools or individual components.

## Models

Our tagger and extractor models are integrated with the Hugging Face library and can be accessed and used in isolation at:

* tagger: https://huggingface.co/IEETA/BioNExt-Tagger
* extractor: https://huggingface.co/IEETA/BioNExt-Extractor

**Authors:**

- Tiago Almeida ([ORCID: 0000-0002-4258-3350](https://orcid.org/0000-0002-4258-3350))
- Richard A A Jonker ([ORCID: 0000-0002-3806-6940](https://orcid.org/0000-0002-3806-6940))
- Rui Antunes ([ORCID: 0000-0003-3533-8872](https://orcid.org/0000-0003-3533-8872))
- João R Almeida ([ORCID: 0000-0003-0729-2264](https://orcid.org/0000-0003-0729-2264))
- Sérgio Matos ([ORCID: 0000-0003-1941-3983](https://orcid.org/0000-0003-1941-3983))
