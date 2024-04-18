import evaluate
import pandas as pd

from benchita.task import Task


class SquadV2Task(Task):

    def evaluate(self, inference):
        # https://huggingface.co/spaces/evaluate-metric/squad_v2
        metric = evaluate.load("squad_v2")

        references = []
        predictions = []

        for elem in inference:
            predictions.append({
                "prediction_text": elem["output"][0]["answers"]["text"][0],
                "id": elem["output"][0]["id"],
                "no_answer_probability": 0.
            })

            references.append({
                "answers": {
                    "answer_start": elem["expected"][0]["answers"]["answer_start"],
                    "text": elem["expected"][0]["answers"]["text"]
                },
                "id": elem["expected"][0]["id"]
            })

        result = metric.compute(references=references, predictions=predictions)
        return result

    def results_summary(self, results):
        exact = results['exact']
        f1 = results['f1']
        total = results['total']

        return pd.DataFrame({
            "Task": [self.task_name],
            "Exact Matching": [exact],
            "F1": [f1],
            "Total": [total]
        })
