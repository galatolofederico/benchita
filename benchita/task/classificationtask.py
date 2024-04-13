from sklearn.metrics import classification_report
import pandas as pd
import os
import enum

from benchita.task import Task

class ClassificationTask(Task):
    @property
    def classes(self):
        raise NotImplementedError
    
    @property
    def unknown_class(self):
        return "<UNKNOWN>"
    
    def _get_class(self, s):
        if s in self.classes:
            return s

        classes_and_ids = [(c, i) for i, c in enumerate(self.classes)]
        classes_and_ids.sort(key=lambda x: len(x[0]), reverse=True)
        for c, i in classes_and_ids:
            if s.startswith(c) or c.startswith(s):
                return c
            
        return self.unknown_class

    def evaluate(self, inference):
        available_classes = self.classes[:]
        available_classes.append(self.unknown_class)

        y_true = []
        y_pred = []

        for elem in inference:
            y_true.append(self._get_class(elem["expected"]))
            y_pred.append(self._get_class(elem["output"]))

        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    def results_summary(self, results):
        accuracy = results["accuracy"]
        f1_macro = results["macro avg"]["f1-score"]
        f1_weighted = results["weighted avg"]["f1-score"]

        return pd.DataFrame({
            "Accuracy": [accuracy],
            "F1 (macro)": [f1_macro],
            "F1 (weighted)": [f1_weighted]
        })



class ClassficationTaskFromCSV(ClassificationTask):
    def __init__(self, *, csv_file, text_column, label_column, labels, read_csv_kwargs={}, join_base_folder=True):
        super().__init__()
        if join_base_folder: csv_file = os.path.join(self.base_folder, csv_file)
        self.df = pd.read_csv(csv_file, **read_csv_kwargs)
                
        assert isinstance(labels, enum.EnumMeta)

        self.text_column = text_column
        self.label_column = label_column
        self.labels = labels

    def _label_to_class(self, label):
        for l in self.labels:
            if label == l.value:
                return l.name
        raise ValueError(f"Invalid label {label}")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        elem = self.df.iloc[idx]
        text = elem[self.text_column]
        label = self._label_to_class(elem[self.label_column])

        return {
            "input": text,
            "output": label
        }
    
    @property
    def classes(self):
        return [l.name for l in self.labels]
    
    @property
    def max_new_tokens(self):
        return max([len(l.name) for l in self.labels])
    
    @property
    def system(self):
        raise NotImplementedError
    
    @property
    def inject_confirmation(self):
        raise NotImplementedError
    
    @property
    def inject_confirmation_reply(self):
        raise NotImplementedError