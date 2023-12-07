### EP

import os
import openai
import numpy as np
import time
import random
import torch
from sentence_transformers import SentenceTransformer, util


import config

random.seed(13)

### initial prompt

##### zero-shot
context = "Speaker: I’ve been hearing some strange noises around the house at night.\n\
    Listener: oh no! That’s scary! What do you think it is?\n\
Speaker: I don’t know, that’s what’s making me anxious.\n\
    Listener:"

prompt_EP_0 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is the existing dialogue context:\n\n" +context

##### few-shot
prompt_EP_1 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is an instance:\n\n"

prompt_EP_5 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is some instances:\n\n"

### prepare for similarity calculation 
data_train_dialog = np.load("data/ED/sys_dialog_texts.train.npy", allow_pickle=True)
data_train_target = np.load("data/ED/sys_target_texts.train.npy", allow_pickle=True)

assert len(data_train_dialog) == len(data_train_target)

datalen = len(data_train_dialog)
print(datalen)

id = 0
conv_data, tmp = [], []
while id < datalen:
    utter = data_train_dialog[id]
    utter.append(data_train_target[id])
    conv_data.append(utter)
    id = id + 1

print(len(conv_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-mpnet-base-v2').to(device)

traindata = []
for conv in conv_data:
    tmp = ""
    for u in conv:
        tmp += u.strip() + " "
    traindata.append(tmp)

traindata_embedding = model.encode(traindata, convert_to_tensor=True, device=device)


### your openai.api_key
openai.api_key = config.apikey

def get_res_chatgpt(m_name, contents):
    response = openai.ChatCompletion.create(
      model=m_name, 
      messages=contents, 
      temperature=config.temperature,
    )['choices'][0]['message']['content']

    return response

def get_conv(m_name, context, prompt_EP_5_sensim):
    context_list = []
    context_list.append({
      'role': 'system',
      'content': prompt_EP_5_sensim
    })

    for i, text in enumerate(context):
        if len(text) == 0:
            continue
        if i % 2 == 0:
            role_str = 'user'
        else:
            role_str = 'assistant'
        context_list.append({
            'role': role_str,
            'content': text
        })

    response = get_res_chatgpt(m_name, context_list)

    return response

def get_maxsim(querydata, k=5):
    testdata_embedding = model.encode(querydata, convert_to_tensor=True, device=device)
    cosine_scores = util.cos_sim(testdata_embedding, traindata_embedding)[0].detach().cpu().numpy()
    sort = np.argsort(cosine_scores)[::-1]
    fewshot_indices, fewshot_sim_scores = sort[:k], [cosine_scores[idx] for idx in sort[:k]]
    # print(fewshot_indices)

    return fewshot_indices

def main():

    fw = open(config.save_path, mode="a")
    data_test = np.load("data/ED/sys_dialog_texts.test.npy", allow_pickle=True)
    len_test = len(data_test)
    # sensims = open("results/sensim_5_ED.txt", mode="r").readlines()

    continueid = -1     # continue after interruption, default to -1
    id = continueid + 1
    while id < len_test:
        testdata = []
        for d in data_test[id]:
            tmp = ""
            for j in d:
                tmp += j.strip() + " "
            testdata.append(tmp)

        sensims = get_maxsim(testdata, 5)
        # print(sensims)
        # print(type(sensims.tolist()))
        Instances = ""
        # for idx, k in enumerate(sensims[id].strip().strip('[').strip(']').split(" ")):
        for idx, k in enumerate(sensims.tolist()):
            # if len(k) > 0:
            #     # print(k)
            #     k = int(k)
            tmp = data_train_dialog[k]
            tmp.append(data_train_target[k].strip())
            Instances += "Instance " + str(idx + 1) + ":\n"
            for i, j in enumerate(tmp):
                if i % 2:
                    Instances += "user: " + j + "\n"
                else:
                    Instances += "assistant: " + j + "\n"

        prompt_EP_5_sensim = prompt_EP_5 + Instances
        # try:
        #     res = get_conv(config.model, data_test[id], prompt_EP_5_sensim)
        # except:
        #     time.sleep(3)
        #     continue

        res = get_conv(config.model, data_test[id], prompt_EP_5_sensim)
    
        fw.write(str(id) + " #*#*# " + res + "\n")
        print(id)
        id = id + 1
    fw.close()

if __name__ == '__main__':
    main()

    ### save as reuse, for saving time
    # fw = open("./sensim_5.txt", mode="w")
    # dialog_test = np.load('data/ED/sys_dialog_texts.test.npy', allow_pickle=True)
    # testdata = []
    # for d in dialog_test:
    #     tmp = ""
    #     for j in d:
    #         tmp += j.strip() + " "
    #     testdata.append(tmp)

    # id = 0
    # len_test = len(testdata)
    # k = 5

    # while id < len_test:
    #     testdata_embedding = model.encode(testdata[id], convert_to_tensor=True, device=device)
    #     cosine_scores = util.cos_sim(testdata_embedding, traindata_embedding)[0].detach().cpu().numpy()
    #     sort = np.argsort(cosine_scores)[::-1]
    #     fewshot_indices, fewshot_sim_scores = sort[:k], [cosine_scores[idx] for idx in sort[:k]]
    #     fw.write(str(fewshot_indices) + '\n')
    #     id = id + 1

    # fw.close()