# sparv-sbx-pi-detection  

A plugin for [Sparv](https://spraakbanken.gu.se/sparv/) for detecting personal information in Swedish texts, especially learner essays (_note: performs noticeably worse on other domains, but the models used for annotation will likely be updated in the future)_.  

**Note that this plugin does not guarantee full accuracy of predictions and should only be used to assist PII annotation or de-identification procedures, with a human verifying the final result.**

## Install

1. Install [Sparv](https://spraakbanken.gu.se/sparv/):  
```
pipx install sparv
sparv setup
```

2. Clone this plugin's repository:  
```
git clone https://github.com/spraakbanken/sparv-sbx-pi-detection
```

3. Inject the plugin into Sparv:  
```
pipx inject sparv /YOUR/PATH/TO/THE/PLUGIN/sparv-sbx-pi-detection
```

Alternatively to 2 & 3:
```
pipx inject sparv https://github.com/spraakbanken/sparv-sbx-pi-detection/archive/main.zip
```

_In case you need to uninstall the plugin:_  
```
pipx runpip sparv uninstall sparv-sbx-pi-detection
```
**NOTE: If you want to use a genLLM to obtain predictions, you must install Ollama!** For instructions on how to do that, look [here](https://ollama.com/download/linux).

## Usage

In your `config.yaml` file for corpus configuration, you can now specify `sbx_pi_detection.pi` as a token annotation alongside one of the 6 annotation types (`basic, basic_iob, general, general_iob, detailed, detailed_iob`). The differences are described in [this publication](https://aclanthology.org/2025.nodalida-1.70/). Effectively, you need to include the following elements in your corpus configuration (substituting `general` for the annotation level you want). You can also specify a HuggingFace model name or local path (e.g. `sbx/KB-bert-swedish_PI-detection-general` in this model's case) to easily access your own models or models not provided in the plugin by default:  

```
export:
  annotations:
    - <token>
    - <token>:sbx_pi_detection.pi
sbx_pi_detection:
  annotation_level: general
```

Alternatively, if you want to use a generative LLM with Ollama for this task, you have to specify three other arguments in the configuration, namely `generative_model`, `prompt_path`, and `batch_size`. The model name should correspond to a model available via Ollama. The prompt path should point to a .txt file with the text of your prompt (see [example prompt file](https://github.com/spraakbanken/sparv-sbx-pi-detection/blob/main/sbx_pi_detection/testprompt/prompt.txt)). The batch size should be an integer that determines how many tokens the model gets to see at once:

```
export:
  annotations:
    - <token>
    - <token>:sbx_pi_detection.pi
sbx_pi_detection:
  generative_model: gemma4:12b
  prompt_path: ../testprompt/prompt.txt
  batch_size: 8
```
Note that you may want to tweak the prompt file to match your use case, e.g. by changing the classes the model should assign to PII entities or providing definitions for said classes. It is highly recommended to keep the first line of the example prompt.

See also the [example corpus config file](https://github.com/spraakbanken/sparv-sbx-pi-detection/blob/main/sbx_pi_detection/testkorpus/config.yaml).

## Models

All of the models used in this plugin are classifiers based on [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased). They can be found [here](https://huggingface.co/collections/Turtilla/pi-detection-and-labeling-6822fbc8fbccfc2527c019ba). See [this publication](https://aclanthology.org/2025.nodalida-1.70/) for training details.  

See the webpage of the [Mormor Karl project](https://mormor-karl.github.io/) for more information on the work we do on automatic pseudonymization of research data.

When using a genLLM with Ollama, you can first explore the models available in its [library](https://ollama.com/library).

## Citing

If you use this plugin e.g. in a publication, please cite it as:

[Språkbanken Text. sbx-swe-pi_detection-sparv [Analysis]. Språkbanken Text. https://doi.org/10.23695/6wp0-ds77](https://spraakbanken.gu.se/en/analyses/sbx-swe-pi_detection-sparv)

And

Maria Irena Szawerna, David Alfter, and Elena Volodina. 2025. [Annotating Personal Information in Swedish Texts with SPARV.](https://aclanthology.org/2025.lm4dh-1.15/) In _Proceedings of the First Workshop on Natural Language Processing and Language Models for Digital Humanities_, pages 155–163, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria.

For the models, cite: 

Maria Irena Szawerna, Simon Dobnik, Ricardo Muñoz Sánchez, and Elena Volodina. 2025. [The Devil’s in the Details: the Detailedness of Classes Influences Personal Information Detection and Labeling](https://aclanthology.org/2025.nodalida-1.70/). In _Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025)_, pages 697–708, Tallinn, Estonia. University of Tartu Library.



