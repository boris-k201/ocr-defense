# Exp3 = untargeted attack (Denial of Answer)

from tqdm import tqdm
from collections import defaultdict
import random
import logging

from .experiment_base import Experiment, Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Exp3(Experiment):
    def __init__(self, n_questions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_questions = n_questions

        self.results = defaultdict(dict)

    def _extract_questions_answers(self, questions_answer_pairs:list):
        questions = []
        answers = []
        for question, answer in questions_answer_pairs:
            questions.append(question)
            answers.append(list(answer)[0]) 
        
        return questions, answers

    def create_adv_examples(self):
        dataset = self.data_loader

        for key, sample in tqdm(dataset.items(), total=len(dataset), desc="Processed"):
            image = sample["image"]
            # extract the GT
            if self.args.n_questions > 0:
                items = random.sample(list(sample["questions"].items()), self.args.n_questions)
            questions, targets = self._extract_questions_answers(items)

            advx = self.attack(model=self.model,
                              processor=self.processor,
                              auto_processor=self.autoprocessor,
                              image=image,
                              questions=questions,
                              targets=targets,
                              is_targeted=False,
                              mask_function=self.mask_function,
                              args=self.args)
            
            assert image.size == advx.size
            
            advx.save(f"{self.res_directory}/{key}.jpg")

            y_pred = self.model.torch_predict(image, questions)
            y_pred_adv = self.model.torch_predict(advx, questions)
            
            assert key not in self.results.keys(), "Found sample in results"
            self.results[key]["questions"] = tuple(questions)
            self.results[key]["gt"] = tuple([sample["questions"][q] for q in questions]) # one question can have multiple answers
            self.results[key]["y_pred"] = tuple(y_pred)
            self.results[key]["y_pred_adv"] = tuple(y_pred_adv)

            print(f'y_pred={y_pred}, y_pred_adv={y_pred_adv}, GT={self.results[key]["gt"]}', flush=True)
        
        self.save_results(self.results)


    def report(self):
        pass