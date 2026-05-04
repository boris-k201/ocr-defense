"""Wrapper for the Donut model."""

import torch
from PIL import Image
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader
from transformers import AutoProcessor, VisionEncoderDecoderModel
from typing import Union, List

import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
logger = logging.getLogger(__name__)

class DonutModelProcessor(DataProcessing):
    """Data processing utility for Donut model."""

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa",
        *,
        local_files_only: bool = False,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    def _process(self, x: Image, q: str) -> torch.Tensor:
        return self.processor(images=x, text=q, return_tensors="pt")

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented."""

    def decode(self, x):
        """Decode model outputs."""
        return self.processor.decode(x, skip_special_tokens=True)

    def get_input_ids(self, text: Union[List,str]):
        # get eos token tensor
        eos_id = self.processor.tokenizer.eos_token_id
        eos = torch.tensor([eos_id])
        
        if isinstance(text, str):
            return self.processor.tokenizer(text=text, return_tensors="pt").input_ids
        elif isinstance(text, list):
            if len(text) > 0:
                tokenized_list = []
                for item in text:
                    item += "</s_answer>"
                    tokenized = self.processor.tokenizer(text=item, 
                                                         add_special_tokens=False,
                                                         return_tensors="pt").input_ids
                    
                    tokenized_list.append(tokenized)
                    tokenized_list.append(torch.tensor([-1], dtype=torch.long).unsqueeze(0))
                
                tokenized_list.pop() # remove the last separator
                res = torch.cat(tokenized_list,dim=1)

                return res
            else:
                raise ValueError("You have to put at least one element") 
        elif text is None:
            return text
        else:
            raise ValueError("`text` has to be either list or str")  
    
    def reconstruct_targets(self, targets, separator=-1): # TODO: put in a shared class
        res = []
        indices = (targets.flatten() == separator).to(dtype=torch.uint8).nonzero()
        
        if indices.numel() == 0: # edge case, just one target
            return targets

        start_index = 0
        for index in indices.squeeze(0):
            res.append(targets[:, start_index:index].squeeze(0))
            start_index = index + 1
        
        res.append(targets[:, index+1:].squeeze(0))

        return res


class DonutModel(BaseModel):
    """Wrapper for the DocVQA PyTorch model."""
    MAX_OUTPUT_TOKENS = 50

    def __init__(
        self,
        device="cpu",
        model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa",
        *,
        local_files_only: bool = False,
    ) -> None:
        self._model: torch.nn.Module = (
            VisionEncoderDecoderModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        ).to(device)

        self.task_prompt = "<s_docvqa><s_question>{question}</s_question><s_answer>"

        self.model_processor = DonutModelProcessor(model_name=model_name, local_files_only=local_files_only)
        self.auto_processor = AutoProcessor.from_pretrained(
            model_name,
#            backend='torchvision',
            use_fast=True,
            local_files_only=local_files_only,
        )
        self._model.config.decoder_start_token_id = self.auto_processor.tokenizer.cls_token_id
        self._model.config.pad_token_id = self.auto_processor.tokenizer.pad_token_id
        super().__init__()

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return prediction from the Pix2Struct model."""
        return self.decision_function(x)[0]

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self._get_device())
        return self._model.generate(flattened_patches=x)

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        Return the gradient of the loss w.r.t. the input.

        Parameters
        ----------
        x : torch.Tensor
            Input patches.
        y : int
            Target answer.

        Returns
        -------
        torch.Tensor
            Gradient of the loss w.r.t. x.
        """
        x = x.to(device=self._get_device())
        x.requires_grad = True
        loss = self.loss_fn(x=x, y=y)
        loss.backward()
        return x.grad


    def loss_fn(self, x: torch.Tensor, target_answer_ids: torch.Tensor, question: str):
        device = self._get_device()
        processor = self.auto_processor

        # tokenize both prompt and target answer
        prompt = self.task_prompt.format(question=question)
        prompt_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

        prompt_ids = prompt_ids.to(device)
        target_answer_ids = target_answer_ids.to(device)
        x = x.to(device)
        
        full_decoder_input_ids = torch.cat([
            prompt_ids, 
            target_answer_ids
        ], dim=1)
        
        outputs = self._model(
            pixel_values=x,
            decoder_input_ids=full_decoder_input_ids
        )
        # Loss computation. Recall that the output has the same shape of the tokens provided as input into the decoder, together with the voc_size and the batch size
        total_loss = outputs.logits.sum() * 0.0
        target_answer_ids = target_answer_ids.squeeze(0)

        for i in range(len(target_answer_ids)):
            logit_position = prompt_ids.shape[1] + -1 + i
            current_logits = outputs.logits[:, logit_position, :] # shape batch, tokens, voc_size
            
            top1_logit, top1_idx = torch.max(current_logits, dim=-1)
            
            target_id = target_answer_ids[i]
            target_logit = current_logits[0, target_id]
            
            if top1_idx.item() != target_id:
                token_loss = top1_logit.squeeze() - target_logit
                total_loss += token_loss
            else:
                total_loss += 0.0        

        # self._debug_token_optimization(x=x,
        #                                total_loss=total_loss,
        #                                target_answer_ids=target_answer_ids,
        #                                prompt_ids=prompt_ids,
        #                                outputs=outputs)

        return outputs, total_loss

    def _debug_token_optimization(self, x, total_loss, target_answer_ids, prompt_ids, outputs):
        processor = self.auto_processor
        device = self._get_device()
        
        self._model.eval()
        with torch.no_grad():
            logger.info(f"###### Debug #####")
            logger.info(f"Loss: {total_loss.item():.4f}")
            
            # check each token in the target sequence, print the top k=5
            k = 5
            logger.info(f'\n top {k} TOKENS')
            for i in range(len(target_answer_ids)):
                logger.info(f"\n Token target n.{i} - {processor.decode(target_answer_ids[i])}")
                logit_position = prompt_ids.shape[1] - 1 + i
                step_logits = outputs.logits[:, logit_position, :]

                probs = torch.nn.functional.softmax(step_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, k=k)

                for j in range(k):
                    token_id = top_k_indices[0, j].item()
                    token_prob = top_k_probs[0, j].item()
                    decoded_token = processor.decode([token_id])
                    logger.info(f"\t\t{j+1}°: token='{decoded_token}', prob={token_prob:.4f}")

            generated_outputs = self._model.generate(
                pixel_values=x.to(device),
                decoder_input_ids=prompt_ids,
                max_length=self.MAX_OUTPUT_TOKENS,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_outputs, skip_special_tokens=False)[0]
            logger.info(f"GENERATED = {generated_text}")
            logger.info("#" * 40)

    def train(self, dataloader: DataLoader) -> BaseModel:
        """Not implemented."""

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input patches.

        Returns
        -------
        torch.Tensor
            Output from the model.
        """
        x = x.to(device=self._get_device())
        return self._decision_function(x)

    def torch_predict(self, image, questions):
        device = self._get_device()
        auto_processor = self.auto_processor
        
        prompts = [self.task_prompt.format(question=q) for q in questions]
        inputs = auto_processor(images=[image]*len(questions),
                                text=prompts, 
                                return_tensors="pt",
                                padding=True).to(device)

        self._model.eval()
        outputs = self._model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            max_length=self.MAX_OUTPUT_TOKENS,
            do_sample=False
        )

        # generate the answer and filter out special tokens
        generated_texts = auto_processor.batch_decode(outputs, skip_special_tokens=False)
        answers = []
        for text in generated_texts:
            clean_text = text.replace(auto_processor.tokenizer.pad_token, "").replace(auto_processor.tokenizer.eos_token, "")

            if "<s_answer>" in clean_text:
                answer_part = clean_text.split("<s_answer>", 1)[1]
                answer = answer_part.split("</s_answer>", 1)[0]
                answers.append(answer.strip())
            else:
                print("*********************** DONUT ERROR")
                print("Answer without <s_answer>")
                print(f"Clean text = {clean_text}")
                print("*********************** DONUT ERROR")
                print("",flush=True)

        return answers
