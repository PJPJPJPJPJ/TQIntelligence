
import sys
import torch
from torch import optim
import numpy as np
from data_0710 import Data_emotion, Label_weight
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import torchmetrics
import itertools
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score

import pdb


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

root = '/Users/pjpjpj/Documents/OMSA/Practicum/TQIntelligence/Data/voice_labeling_report_21_May_22'
json_path = 'voice_labels_s.json'
root_wav = '/Users/pjpjpj/Documents/OMSA/Practicum/TQIntelligence/Data/voice_samples/'

json_file_path = os.path.join(root, json_path)
with open(json_file_path) as json_file:
    data = json.load(json_file)

weights = Label_weight(data, ['Pegah Moghaddam','Michelle Lyn','Sedara Burson','Yared Alemu'])
weights_onedoc = weights.get_weight()

data_length = len(data)
l_train = int(np.floor(data_length * 0.3))
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

model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

#output_class = 20
# model.lm_head = torch.nn.Sequential(torch.nn.Linear(1024, 100), torch.nn.Flatten(0,-1), torch.nn.ReLU(), torch.nn.Linear(284300, 100))
model.lm_head = torch.nn.Sequential(torch.nn.Flatten(0,-1), torch.nn.Linear(799744, 100))
model = model.to(device)

# Change it to adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.002) # ADAM optimizer (Most well-known optimizer for deep learning)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, min_lr=1e-5, verbose=True) # Learning rate schedular which controls the learning rate during training procedure

# forward
epochs = 100
best_p = 0
for epoch in range(0, epochs):
    model.train()
    mean_loss = []
    with tqdm(total=int(n_train*batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for input, labels in trainloader:
            input, labels =  input.to(device), labels.to(device)
            #input = input.half() # Float 32- 16
            batch_size = input.shape[0]
            input = (input-torch.min(input))/(torch.max(input)-torch.min(input))

            if labels.shape[1]==1:
                continue

            # Set the gradient in the model into 0
            optimizer.zero_grad()
            #with torch.cuda.amp.autocast():
            prediction = model(input) # returns: (batch_size, time_step, feature_size)

            prediction = prediction.logits.view(5, 20)
            final_prediction = torch.nn.Softmax(dim = 1)(prediction)

            ####calculate loss funtion####
            loss = torch.nn.CrossEntropyLoss(weight = weights_onedoc)(final_prediction, labels.squeeze().long())
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
            input, labels =  input.to(device), labels.to(device)
            #input = input.half()
            input = (input-torch.min(input))/(torch.max(input)-torch.min(input))

            if labels.shape[1]==1:
                continue

            # Set the gradient in the model into 0
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                prediction = model(input).logits # returns: (batch_size, time_step, feature_size)
            pdb.set_trace()
            prediction = prediction.reshape(5, 20)
            final_prediction = torch.nn.Softmax(dim=1)(prediction)
            loss = torch.nn.CrossEntropyLoss(weight = weights_onedoc)(final_prediction, labels.squeeze().long()) # Loss function measures the difference between predictions and labels
            pdb.set_trace()
            # pred_label = []
            # for i in [0,1,2,3,4]:
            #     single_emo_block = final_prediction[:,i*4:i*4+4]
            #     single_emo_max = torch.argmax(single_emo_block)
            #     single_emo_max = i*4 + (single_emo_max.item()) % 4
            #     pred_label.append(single_emo_max)

            # pred_label = torch.Tensor(pred_label)

            final_prediction = final_prediction.reshape(20,5)

            prediction_sort, label = torch.sort(final_prediction, 1, descending = True)
            
            max_prob = prediction_sort[0,:]
            max_prob_sort, index_max_prob = torch.sort(max_prob, descending=True)

            prediction_sort2 = prediction_sort[:,index_max_prob]
            label2 = label[:,index_max_prob]
            

            decision_l = [label2[0,0].item()]
            pdb.set_trace()

            for i in range(1,prediction_sort2.shape[0]):
                for j in range(label2.shape[0]):
                    if torch.all(abs(torch.Tensor(decision_l)-label2[j,i])>3):
                        decision_l.append(label2[j,i].item())
                        temp_label = label2[j,i]
                        break
            #pdb.set_trace()
            pred_label = torch.Tensor(decision_l)


            # max_label = torch.argmax(final_prediction)
            # pred_label = [0,0,0,0]
            # pred_label[max_label.item()] = 1

            pred = pred_label.detach().cpu().numpy().astype(np.uint8)
            labels = labels.squeeze().detach().cpu().numpy().astype(np.uint8)
            print(pred)
            print(labels)

            precision = precision_score(pred, labels, average='macro')
            recall = recall_score(pred, labels, average='macro')
            f1 = f1_score(pred, labels, average='macro')

            print('p: {}, r:{}, f1:{}'.format(precision, recall, f1))
            p_lst.append(precision)
            r_lst.append(recall)
            f1_lst.append(f1)

            # Update the pbar
            pbar.update(1)

            # Add loss (batch) value to tqdm
            pbar.set_postfix(**{'val_CE_loss': loss.item()})

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
