from copy import deepcopy
from typing import Optional
from transformers import T5Tokenizer, T5Model, T5TokenizerFast
import torch
import numpy as np


from ..nn_classifier import BaseNNClassifier
from .base import BaseClassifier

class LangCondClassifier(BaseClassifier):
    def __init__(
            self,
            nn_classifier: BaseNNClassifier,
            device: str = "cpu",
            optim_params: Optional[dict] = None,
    ):
        super().__init__(nn_classifier, 0.995, None, optim_params, device)
        # initializes language encoder
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-base")
        self.text_model = T5Model.from_pretrained("t5-base").to(device)
        self.input_ids = self.tokenizer(
            "", return_tensors="pt"
        ).input_ids.to(device)  # Batch size 1

        # we need to have a little MLP reduce the condition dimension from 
        # 786 to something smaller so we just do that more or less manually here
        # using buildling the MLP in cleandiffuser's MLP section
        in_dim = 768
        out_dim = 64
        activation = torch.nn.SiLU()
        hidden_dims = (768,768)
        self.condition_mlp = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    activation,
                )
                for i in range(len(hidden_dims))
            ],
            torch.nn.Linear(hidden_dims[-1], out_dim)
        )
        self.condition_mlp.to(self.device)

    def loss(self, x, noise, lang):

        trajectory = x # just to make formating nicer for me
        lang_encoding = torch.squeeze(self.get_language_encoding(lang))

        # makes fake lang encoding where index is shuffled by 1 (with wrap around)
        # this guarantees no collisions
        lang_encoding_fake = torch.zeros_like(lang_encoding)
        lang_encoding_fake[1:] = lang_encoding[:-1]
        lang_encoding_fake[0] = lang_encoding[-1]
        

        #print(lang_encoding.shape)
        #print(lang_encoding[:10,0])
              
        #print(lang_encoding_fake[:10,0])

        # passes both language encoding things through MLPs


        
        #lang_encoding = self.condition_mlp(lang_encoding)
        #lang_encoding_fake = self.condition_mlp(lang_encoding_fake)
        # passes through model to get logit predictions
        ##print(trajectory.shape)
        #print(noise.shape)
        #print(lang_encoding.shape)
        pred_true = self.model(trajectory, noise, lang_encoding)
        pred_false = self.model(trajectory, noise, lang_encoding_fake)

        #print(pred_true.shape)
        
        # gets loss of true and fake batches
        loss_function = torch.nn.BCEWithLogitsLoss()
        #loss_function = torch.nn.MSELoss()
        bce_logit_loss_true = loss_function(pred_true,torch.ones_like(pred_true))
        bce_logit_loss_false = loss_function(pred_false,torch.zeros_like(pred_false))

        # averages
        final_loss = (bce_logit_loss_true + bce_logit_loss_false).mean()
        #print(pred_true[0].item(),pred_false[0].item())
        print(final_loss)
        #print(pred_false)
        return final_loss

    def update(self, x, noise, lang):
        self.optim.zero_grad()
        loss = self.loss(x, noise, lang)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c):
        return self.model_ema(x, noise, c)

    def get_language_encoding(self,text_block):
        encoding_storage = []
        for text in text_block:
            decoder_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)  # Batch size 1
            outputs = self.text_model(input_ids=self.input_ids, decoder_input_ids=decoder_input_ids)
            last_hidden_states = torch.mean(outputs.last_hidden_state,dim=1)
            encoding_storage.append(last_hidden_states.detach())
        
        return_tensor = torch.stack(encoding_storage)
        return return_tensor
