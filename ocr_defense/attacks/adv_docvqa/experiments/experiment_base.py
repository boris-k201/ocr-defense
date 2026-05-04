import random
import numpy as np
import torch
import os
from abc import ABC, abstractmethod
import editdistance
from transformers import set_seed      

class Metrics():
    def __init__(self):
        self.asnlr = 0.0
        self.asr = 0.0
        self.cdmg = 0.0
        self.total_accuracies = []
        self.total_anls = []
        self.anls_threshold = 0.5
        self.get_edit_distance = editdistance.eval
    
    def update_anls(self, gt, pred, return_anls=False):
        # ref: https://github.com/rubenpt91/MP-DocVQA-Framework/blob/master/metrics.py
        if len(pred) == 0:
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        if return_anls==False:
            self.total_anls.append(anls)
        else:
            return anls

    def update_batch_anls(self, gt_batch, pred_batch):
        anls = 0

        for gt, pred in zip(gt_batch, pred_batch):
            anls += self.update_anls(gt, pred, return_anls=True)
        
        anls /= len(gt_batch)
        self.total_anls.append(anls)

    def update_accuracy(self, gt:set, pred:str):
        if pred in gt:
            self.total_accuracies.append(1)
        else:
            self.total_accuracies.append(0)
    
    def build_results_key(self, image_id, question):
        return f"{image_id}.{question[:50]}"
    
    def get_accuracy(self):
        return sum(self.total_accuracies) / len(self.total_accuracies) if len(self.total_accuracies) > 0 else 0.0

    def get_anls(self):
        return sum(self.total_anls) / len(self.total_anls) if len(self.total_anls) > 0 else 0.0


class Experiment(ABC):
    RESULTS_FOLDER = 'results'

    def __init__(self, attack, model, autoprocessor, processor, data_loader, mask_function, args):
        self.attack = attack
        self.model = model
        self.autoprocessor = autoprocessor
        self.processor = processor
        self.data_loader = data_loader
        self.mask_function = mask_function
        self.args = args

    def set_seed(self, seed:int) -> None:  
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_seed(seed)

    def setup(self):
        self.set_seed(self.args.seed)

        # Prepare the results' directory
        self.res_directory = (f'{Experiment.RESULTS_FOLDER}/{self.__class__.__name__}/{self.args.n_questions}-' +
                              f'{self.args.mask}-{self.model.__class__.__name__}/')
        if not os.path.exists(self.res_directory):
            os.makedirs(self.res_directory)

    def save_results(self, results):
        import pickle
        filename = f"doa_results_{self.model.__class__.__name__}.pkl"
        results_path = os.path.join(self.res_directory, filename)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    @abstractmethod
    def create_adv_examples(self):
        pass

    @abstractmethod
    def report(self):
        pass