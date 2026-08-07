from src.processor import Processor
from typing import List

class Pipeline:
    def __init__(self):
        self.steps: List[Processor] = []

    def add_step(self, step):
        self.steps.append(step)

    def process_all(self, kwargs):
        result_total = []
        metrics = {}
        for step in self.steps:
            # print('Processando step:', )
            # print('Textos:', kwargs['text1'], kwargs['text2'])
            (text1, text2), eval, step_metrics = step.apply_pairs(**kwargs)
            result_total.append(eval)
            metrics.update(step_metrics)
            kwargs['text1'] = text1
            kwargs['text2'] = text2
            
            # print(f"   EVAL: {eval}")
        return kwargs, all(result_total), metrics
    def process(self, kwargs):
        result_total = []
        for step in self.steps:
            # print('Processando step:', )
            (text1, text2), eval, _ = step.apply_pairs(**kwargs)
            result_total.append(eval)
            if(not eval):
                break

            kwargs['text1'] = text1
            kwargs['text2'] = text2
            
            # print(f"   EVAL: {eval}")
        return kwargs, all(result_total)