
import sys
import torch
from torch import optim
import numpy as np
from data import Data_emotion
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import torchmetrics
import itertools
import os
import json
# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from torch.cuda.amp import autocast as autocast
from model_emo import EmotionModel
from data import Label_weight

import log
import pdb

# Choose the GPUs
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

root = '/lrde/home2/ychen/s3prl'
json_path = 'voice_labels.json'
root_wav = '/lrde/image/voice/voice_samples'


json_file_path = os.path.join(root, json_path)
with open(json_file_path) as json_file:
    data = json.load(json_file)

weights = Label_weight(data, ['Pegah Moghaddam','Michelle Lyn','Sedara Burson','Yared Alemu'])
weights_onedoc = weights.get_weight()
weights_onedoc = weights_onedoc.cuda()

data_length = len(data)
l_train = int(np.floor(data_length * 0.5))
l_val =  int(np.floor(data_length * 0.2))
l_test = data_length - l_train - l_val

train_data = dict(itertools.islice(data.items(), 0, l_train))
val_data = dict(itertools.islice(data.items(), l_train, l_train+l_val))
test_data = dict(itertools.islice(data.items(), l_train+l_val, data_length))

batch_size = 1

dm_train = Data_emotion(train_data, root_wav)
trainloader = torch.utils.data.DataLoader(dm_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
n_train = len(trainloader) # The length of the samples

dm_val = Data_emotion(val_data, root_wav)
valloader = torch.utils.data.DataLoader(dm_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
n_val = len(valloader) # The length of the samples

dm_test = Data_emotion(test_data, root_wav)
testloader = torch.utils.data.DataLoader(dm_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
n_test = len(testloader) # The length of the samples

model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
model = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)
model.classifier = torch.nn.Linear(1024, 100)

output_class = 20

model = model.cuda()

# Change it to adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0002) # ADAM optimizer (Most well-known optimizer for deep learning)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, min_lr=1e-5, verbose=True) # Learning rate schedular which controls the learning rate during training procedure

logger = log.get_logger('speech.txt')

# forward
epochs = 100
best_p = 0
for epoch in range(0, epochs):
    model.train()
    mean_loss = []
    with tqdm(total=int(n_train*batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for input, labels in trainloader:
            input, labels =  input.cuda(), labels.cuda()

            batch_size = input.shape[0]
            input = (input-torch.min(input))/(torch.max(input)-torch.min(input))

            if labels.shape[1]==1:
                continue

            # Set the gradient in the model into 0
            optimizer.zero_grad()

            prediction = model(input) # returns: (batch_size, time_step, feature_size)

            prediction = prediction[1].reshape(5, 20)
            final_prediction = torch.nn.Softmax(dim=0)(prediction)

            ####calculate loss funtion####
            loss = torch.nn.CrossEntropyLoss(weight=weights_onedoc)(final_prediction, labels.squeeze().long())
            loss.backward() # BP
            optimizer.step() # Updating the loss weight

            # Update the pbar
            pbar.update(batch_size)

            # Add loss (batch) value to tqdm
            pbar.set_postfix(**{'train_CE_loss': loss.item()})
    model.eval()
    mean_loss = []
    p_lst = []
    r_lst = []
    f1_lst = []
    with tqdm(total=int(n_val*batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for input, labels in valloader:
            input, labels =  input.cuda(), labels.cuda()

            input = (input-torch.min(input))/(torch.max(input)-torch.min(input))

            if labels.shape[1]==1:
                continue

            # Set the gradient in the model into 0
            with torch.no_grad():
                prediction = model(input)[1] # returns: (batch_size, time_step, feature_size)

            prediction = prediction.reshape(5, 20)
            final_prediction = torch.nn.Softmax(dim=1)(prediction)

            loss = torch.nn.CrossEntropyLoss(weight=weights_onedoc)(final_prediction, labels.squeeze().long()) # Loss function measures the difference between predictions and labels

            pred_label = []
            for i in [0,1,2,3,4]:
                single_emo_block = final_prediction[:,i*4:i*4+4]
                single_emo_max = torch.argmax(single_emo_block)
                single_emo_max = i*4 + (single_emo_max.item()) % 4
                pred_label.append(single_emo_max)

            pred_label = torch.Tensor(pred_label)

            pred = pred_label.detach().cpu().numpy().astype(np.uint8)
            labels = labels.squeeze().detach().cpu().numpy().astype(np.uint8)

            precision, recall, fscore, support = score(pred, labels, average='macro', zero_division=0)

            p_lst.append(precision)
            r_lst.append(recall)
            f1_lst.append(fscore)

            # Update the pbar
            pbar.update(1)

            # Add loss (batch) value to tqdm
            pbar.set_postfix(**{'val_CE_loss': loss.item(), 'p': round(precision, 4), 'r': round(recall, 4), 'f1': round(fscore, 4)})

    logger.info('Precision %f, Recall: %f, F1: %f' %
                (torch.from_numpy(np.array(np.mean(p_lst))).cuda(),
                    torch.from_numpy(np.array(np.mean(r_lst))).cuda(),
                    torch.from_numpy(np.array(np.mean(f1_lst))).cuda()))

    print('Total dataset precision {}'.format(np.mean(p_lst)))
    print('Total dataset recall {}'.format(np.mean(r_lst)))
    print('Total dataset f1 {}'.format(np.mean(f1_lst)))

    p_lst = torch.Tensor(p_lst)
    r_lst = torch.Tensor(r_lst)
    f1_lst = torch.Tensor(f1_lst)

    mean_p = torch.mean(p_lst).item()
    mean_r = torch.mean(r_lst).item()
    mean_f1 = torch.mean(f1_lst).item()

    print('Validation Pre: {}'.format(mean_p))
    print('Validation Rec: {}'.format(mean_r))
    print('Validation f1: {}'.format(mean_f1))

    scheduler.step(mean_f1)

    if mean_f1 > best_p:
        best_loss = mean_f1
        torch.save(model.state_dict(), 'best_val.pth')  # Save and update best weight
