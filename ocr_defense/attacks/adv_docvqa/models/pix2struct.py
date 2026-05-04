"""Wrapper for the Pix2Struct model."""

import torch
from PIL import Image
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from typing import Union, List

class Pix2StructModelProcessor(DataProcessing):
    """Data processing utility for Pix2Struct model."""

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")

    def _process(self, x: Image, q: str) -> torch.Tensor:
        return self.processor(images=x, text=q, return_tensors="pt")

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented."""

    def decode(self, x):
        """Decode model outputs."""
        return self.processor.decode(x, skip_special_tokens=True)

    def get_input_ids(self, text: Union[List,str]):
        """Get tokenizer indexes from string."""
        if isinstance(text, str):
            return self.processor.tokenizer(text=text, return_tensors="pt").input_ids
        elif isinstance(text, list):
            if len(text) > 0:
                tokenized_list = []

                # TODO: improve, it's not needed the for loop
                for item in text:
                    tokenized = self.processor.tokenizer(text=item, return_tensors="pt").input_ids
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
    
    def reconstruct_targets(self, targets, separator=-1):
        res = []
        indices = (targets.flatten() == separator).to(dtype=torch.uint8).nonzero()
        
        if indices.numel() == 0: # edge case, just one target
            return targets

        start_index = 0
        for index in indices.squeeze(0):
            res.append(targets[:, start_index:index].squeeze(0))
            start_index = index + 1
        
        res.append(targets[:, index+1:].squeeze(0))
        for i in range(len(res)):
            assert res[i][-1] == self.processor.tokenizer.eos_token_id

        return res


class Pix2StructModel(BaseModel):
    """Wrapper for the DocVQA PyTorch model."""
    MAX_OUTPUT_TOKENS = 50

    def __init__(self, device="cpu") -> None:
        self._model: torch.nn.Module = (
            Pix2StructForConditionalGeneration.from_pretrained(
                "google/pix2struct-docvqa-base",
            )
        ).to(device)
        self.model_processor = Pix2StructModelProcessor()
        super().__init__()

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return prediction from the Pix2Struct model."""
        return self.decision_function(x)[0]

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented"""

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

    def loss_fn(self, x: torch.Tensor, y: str):
        """
        Compute loss function.

        Parameters
        ----------
        x : torch.Tensor
            Input patches.
        y : str
            Target answer.

        Returns
        -------
        predictions: torch.Tensor
            The predictions from the model.
        loss: torch.Tensor
            The loss.
        """
        x = x.to(device=self._get_device())
        y = y.to(device=self._get_device())
        predictions = self._model(
            labels=y,
            flattened_patches=x,
        )
        return predictions, predictions.loss

    def train(self, dataloader: DataLoader) -> BaseModel:
        """Not implemented."""
    
    def torch_predict(self, image, questions):
        device = self._get_device()
        auto_processor = self.model_processor.processor
        inputs = auto_processor(images=[image]*len(questions), text=questions, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # autoregressive generation
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.MAX_OUTPUT_TOKENS)
        generated_texts = auto_processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_texts