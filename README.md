# benchita

High quality benchmarks for Italian LLMs.

## Implemented tasks

| Task      | Source                     | Description                                                                            | Link                                                 | Quality |
|-----------|----------------------------|----------------------------------------------------------------------------------------|------------------------------------------------------|---------|
| sentipolc | sentipolc@<br>Evalita 2014 | Binary classification task extracted from the sentipolc@<br>Evalita 2014 gold test set | http://www.di.unito.it/~tutreeb/sentipolc-evalita14/ | gold    |
| ironita   | Evalita 2018               | Binary classification task of tweets with or without irony                             | http://twita.di.unito.it/dataset/ironita             | gold    |

## installation

```
pip install -e git+https://github.com/galatolofederico/benchita.git
```

## Usage

`benchita evaluate --model <huggingface-model-name> --task <task-name>`