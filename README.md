# benchita

High quality few-shots benchmarks for Italian LLMs.

## Features 

- ✅ **Easy to Use**: Straightforward setup and operation.
- 🎨 **Chat Template Support**: Integrates smoothly with 🤗 chat templates.
- 🛠️ **Custom Template Support**: Allows for personalization with user-defined templates.
- 🗨️ **Versatile Evaluation**: Capable of assessing both **pretrained** and **chat** models

## Implemented tasks

| Task      | Type           | Source          | Description                                                                            | Link                                                 | Quality |
|-----------|----------------|-----------------|----------------------------------------------------------------------------------------|------------------------------------------------------|---------|
| sentipolc | classification | Evalita 2014    | Binary classification task of tweets extracted from the sentipolc@Evalita 2014 gold test set | http://www.di.unito.it/~tutreeb/sentipolc-evalita14/ | gold    |
| ironita   | classification | Evalita 2018    | Binary classification task of tweets with or without irony                             | http://twita.di.unito.it/dataset/ironita             | gold    |

## Installation

To install Benchita, use the following command:

```
pip install -e git+https://github.com/galatolofederico/benchita.git
```

## Usage

Run Benchita with your chosen model and task:

```
benchita --model <huggingface-model-name> --task <task-name>
```

## License

benchita is released under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.en.html). This license allows users to run, study, share, and modify the software, ensuring that all derivatives remain free and open under the same terms.