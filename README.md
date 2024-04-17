# benchita

High quality few-shots benchmarks for Italian LLMs.

## Features 

- 🚀 **Parallel Execution**: Parallel distributes jobs (model+task) across all available GPUs.
- ✅ **Easy to Use**: Straightforward setup and operation.
- 🎨 **Chat Template Support**: Integrates smoothly with 🤗 chat templates.
- 🛠️ **Custom Template Support**: Easily implement user-defined templates.
- 🗨️ **Versatile Evaluation**: Capable of assessing both **pretrained** and **chat** models

## Implemented tasks

| Task      | Type           | Source          | Description                                                                                  | Link                                                                     | Quality |
|-----------|----------------|-----------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---------|
| sentipolc | classification | Evalita 2014    | Binary classification task of positive or negative tweets (gold test set)                    | http://www.di.unito.it/~tutreeb/sentipolc-evalita14/                     | gold    |
| ironita   | classification | Evalita 2018    | Binary classification task of irnoic tweets (gold test set)                                  | http://twita.di.unito.it/dataset/ironita                                 | gold    |
| ami2020   | classification | Evalita 2020    | Binary classification task of misogynous tweets (gold test set)                              | https://live.european-language-grid.eu/catalogue/corpus/7005             | gold    |

## Roadmap

### Features

- [ ] Inference on multiple GPUs with 🤗 accelerate
- [ ] Support for `int8` and other quantization techniques
- [ ] Support for 🤗 peft adapters 

### Dataset
- [ ] squad-it
- [ ] xcopa-it

## Installation

To install Benchita, clone this repository

```
git clone https://github.com/galatolofederico/benchita.git
cd benchita
```

Create and activate a new virtual environment (or use your preferred way)

```
python3 -m venv env
source ./env/bin/activate
```

Install Cenchita

```
pip3 install -e .
```

Also install `PyTorch` in your preferred way, for example:

```
pip3 install torch
```

## Usage

To run Benchita you need to create a YAML config file (default config: `config.yaml`).


For example to compare `cerbero-7b` and `cerbero-7b-openchat` using the EVALITA benchmarks the config will look like this:

```yaml
experiment: cerbero-7b

tasks:
  - name: ironita
    num_shots: 3
  - name: sentipolc
    num_shots: 3
  - name: ami2020
    num_shots: 3

models:
  - model:
      name: galatolo/cerbero-7b
      class: AutoModelForCausalLM
      dtype: float16
    tokenizer:
      max_length: 1024
    template:
      system_style: inject
    generate:
      batch_size: 16
      args:
        do_sample: False
  - model:
      name: galatolo/cerbero-7b-openchat
      class: AutoModelForCausalLM
      dtype: float16
    tokenizer:
      max_length: 1024
    template:
      system_style: inject
    generate:
      batch_size: 16
      args:
        do_sample: False
```

To start the evaluation simply run

```
benchita
```


## License

benchita is released under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.en.html). This license allows users to run, study, share, and modify the software, ensuring that all derivatives remain free and open under the same terms.