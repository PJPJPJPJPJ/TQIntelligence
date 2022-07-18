
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from data import Data_emotion

import torch
import numpy as np
from data import Data_emotion
from transformers import Wav2Vec2Processor
import itertools
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import autocast as autocast

import pdb

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class model_all(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        model = Wav2Vec2Processor.from_pretrained(model_name)
        model = EmotionModel.from_pretrained(model_name)
        model.classifier = torch.nn.Linear(1024, 100)
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return logits[1]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input, labels = train_batch
        input = input.type_as(input)
        labels = labels.type_as(labels)

        input = (input-torch.min(input))/(torch.max(input)-torch.min(input))
        prediction = self.model(input) # returns: (batch_size, time_step, feature_size)
        prediction = prediction[1].reshape(5, 20)
        final_prediction = torch.nn.Softmax(dim=0)(prediction)
        loss = torch.nn.CrossEntropyLoss()(final_prediction, labels.squeeze().long())
        # return {"loss": loss, "pred": final_prediction}
        return loss

    # def training_step_end(self, batch_parts):
    #     # predictions from each GPU
    #     predictions = batch_parts["pred"]
    #     # losses from each GPU
    #     losses = batch_parts["loss"]

    #     gpu_0_prediction = predictions[0]
    #     gpu_1_prediction = predictions[1]

    #     # do something with both outputs
    #     return (losses[0] + losses[1]) / 2

    def validation_step(self, val_batch, batch_idx):
        input, labels = val_batch
        input = (input-torch.min(input))/(torch.max(input)-torch.min(input))
        if labels.shape[1]==1:
            loss = 0
            return loss
        prediction = self.model(input) # returns: (batch_size, time_step, feature_size)
        prediction = prediction[1].reshape(5, 20)
        final_prediction = torch.nn.Softmax(dim=0)(prediction)
        loss = torch.nn.CrossEntropyLoss()(final_prediction, labels.squeeze().long())
        return loss

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.bacward()

root = '/lrde/home2/ychen/s3prl'
json_path = 'voice_labels.json'
root_wav = '/lrde/image/voice/voice_samples'

json_file_path = os.path.join(root, json_path)
with open(json_file_path) as json_file:
    data = json.load(json_file)

data_length = len(data)
l_train = int(np.floor(data_length * 0.3))
l_val =  int(np.floor(data_length * 0.2))
l_test = data_length - l_train - l_val

train_data = dict(itertools.islice(data.items(), 0, l_train))
val_data = dict(itertools.islice(data.items(), l_train, l_train+l_val))
test_data = dict(itertools.islice(data.items(), l_train+l_val, data_length))

# data
dm_train = Data_emotion(train_data, root_wav)
trainloader = torch.utils.data.DataLoader(dm_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
n_train = len(trainloader) # The length of the samples

dm_val = Data_emotion(val_data, root_wav)
valloader = torch.utils.data.DataLoader(dm_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
n_val = len(valloader) # The length of the samples

all = model_all()

# training
trainer = pl.Trainer(gpus=-1, num_nodes=0, precision=16, limit_train_batches=0.5)
trainer.fit(all, trainloader, valloader)

