from sklearn.metrics import classification_report

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

    def evaluate(self, inference_inputs, inference_outputs):
        available_classes = self.classes[:]
        available_classes.append(self.unknown_class)

        y_true = []
        y_pred = []

        for input, output in zip(inference_inputs, inference_outputs):
            true_class = input["expected"]
            y_true.append(self._get_class(true_class))
            y_pred.append(self._get_class(output))

        return classification_report(y_true, y_pred)