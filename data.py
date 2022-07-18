from torch.utils import data
import os
import json
import librosa
import torch
from itertools import permutations
import itertools
import sklearn
import pdb
import inquirer


# questions = [
#     inquirer.List('doctor',
#                 message="Which doctor do you need?",
#                 choices=['Pegah Moghaddam','Michelle Lyn','Sedara Burson','Yared Alemu'],
#             ),
#     # inquirer.List('emotion',
#     #             message="Which emotion do you need?",
#     #             choices=['fear','anger','happy','neutral','sadness'],
#     #         )
# ]
# answers = inquirer.prompt(questions)
# doctor_name = answers['doctor']

doctor_name = 'Michelle Lyn'
#emotion_input = answers['emotion']
print ("Train the model based on Dr. {}'s scores.".format(doctor_name))

#sr = 22050
#doctor_name = 'Michelle Lyn'


class Label_weight:
    def __init__(self, label_file, doctors):
        self.label_file = label_file
        self.doctors = doctors
    
    def get_weight(self):
        emo_dict_bydoc_nofile = {i: {'emo_by_lvl':{'fear':{'none':0,'low':0,'medium':0,'high':0},
                                          'anger':{'none':0,'low':0,'medium':0,'high':0},
                                          'happy':{'none':0,'low':0,'medium':0,'high':0},
                                          'neutral':{'none':0,'low':0,'medium':0,'high':0},
                                          'sadness':{'none':0,'low':0,'medium':0,'high':0}},
                             'total':0} for i in self.doctors}

        for filename in self.label_file:
            for doc in self.label_file[filename]:
                emos = self.label_file[filename][doc]
                for emo in emos:
                    level = emos[emo]
                    emo_dict_bydoc_nofile[doc]['emo_by_lvl'][emo][level] += 1
            emo_dict_bydoc_nofile[doc]['total'] += 1

        emo_coded = {}     
        for doc in emo_dict_bydoc_nofile:
            total = emo_dict_bydoc_nofile[doc]['total']
            emos = emo_dict_bydoc_nofile[doc]['emo_by_lvl']
            temp = []
            for emo in emos:
                for level in emos[emo]:
                    temp.append(emos[emo][level]/total)
            emo_coded[doc] = temp

        return torch.Tensor(emo_coded[doctor_name])

class Data_emotion(data.Dataset):
    # def __init__(self, root, json_file, root_wav):
    def __init__(self, data, root_wav):
        #self.root = root
        self.root_wav = root_wav
        #self.json_file = json_file
        #self.cache = {}
        #json_file_path = os.path.join(self.root, self.json_file)

        #with open(json_file_path) as json_file:
        #    data = json.load(json_file)
        self.wav_files = list(data.keys())
        self.emo_files = list(data.values())
        #self.emo_lvl = self.emo_files[doctor_name][emotion_input]
            # for file in self.wav_files:
            #     if '.m4a' in file:
            #         file = file.replace('.m4a', '.wav')

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index):

        #emotion_ind_dict = {'fear':1, 'anger':2, 'happy':3, 'neutral':4, 'sadness':5}
        wav_file = os.path.join(self.root_wav, self.wav_files[index]).replace('.m4a', '.wav')
        emo_label = self.emo_files[index]
        
        max_len = 910000
        y, y_sr = librosa.load(wav_file, sr=16000)    # Load the wav_file
        y = librosa.util.fix_length(y, size=max_len)
        audio = torch.Tensor(y)

        # for doctor, list_of_emos in emo_label.items():
        #     if doctor == doctor_name:
        #         chosen_emo_lvl = list_of_emos[emotion_input]
        #         return audio, chosen_emo_lvl
        #     else:
        #         return audio, torch.Tensor([-1])
        
        emo_list   = ['fear', 'anger', 'happy', 'neutral', 'sadness']
        emo_lvl_list = ['none', 'low', 'medium', 'high']
        emo_val_list = [0, 1, 2, 3, 4]
        emo_c_list_num = [0, 1, 2, 3]
        emo_comb_list = list(itertools.product(emo_val_list, emo_c_list_num))
        label_dict = {d: None for d in list(emo_label.keys())}
        label_list = [-1]
        for doctor, value in emo_label.items():
            if doctor == doctor_name:
                doc_list = []
                for index, (i, j) in enumerate(value.items()):
                    emo_index = emo_list.index(i)
                    emo_lvl_index = emo_lvl_list.index(j)
                    emo_index = (emo_index, emo_lvl_index)
                    emo_label_all = emo_comb_list.index(emo_index)
                    doc_list.append(emo_label_all)
                # Change emo_strength into tensor
                label_list = torch.Tensor(doc_list)
        if len(label_list)>1:
            # audio = (audio-min(audio))/(max(audio)-min(audio))
            return audio, label_list
        else:
            return audio, torch.Tensor([-1])


if __name__ == '__main__':
    root = '/Users/pjpjpj/Documents/OMSA/Practicum/TQIntelligence/Data/voice_labeling_report_21_May_22'
    json_path = 'voice_labels_s.json'
    root_wav = '/Users/pjpjpj/Documents/OMSA/Practicum/TQIntelligence/Data/voice_samples/'
    dm = Data_emotion(root, json_path, root_wav)
    print(dm[0])
    pdb.set_trace()