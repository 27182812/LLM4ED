### ED

import os
import openai
import numpy as np
import time
import random

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

data_train_dialog = np.load("data/ED/sys_dialog_texts.train.npy", allow_pickle=True)
data_train_target = np.load("data/ED/sys_target_texts.train.npy", allow_pickle=True)

len_train = len(data_train_dialog)
shot1 = random.randint(0, len_train-1)
shot5 = random.sample(range(0,len_train), 5)

tmp = data_train_dialog[shot1]
tmp.append(data_train_target[shot1].strip())

### for 1-shot
Instance = "Instance:\n"
for i, j in enumerate(tmp):
        if i % 2:
                Instance += "user: " + j + "\n"
        else:
                Instance += "assistant: " + j + "\n"

prompt_EP_1 = prompt_EP_1 + Instance

### for 5-shot
Instances = ""
for id, k in enumerate(shot5):
    tmp = data_train_dialog[k]
    tmp.append(data_train_target[k].strip())
    Instances += "Instance " + str(id+1) + ":\n"
    for i, j in enumerate(tmp):
        if i % 2:
            Instances += "user: " + j + "\n"
        else:
            Instances += "assistant: " + j + "\n"

prompt_EP_5 = prompt_EP_5 + Instances

### your openai.api_key
openai.api_key = config.apikey

def get_res_chatgpt(m_name, contents):
    response = openai.ChatCompletion.create(
      model=m_name, 
      messages=contents, 
      temperature=config.temperature,
    )['choices'][0]['message']['content']

    return response

def get_conv(m_name, context, few_shot=1):

    if few_shot == 1:
         prompts = prompt_EP_1
    elif few_shot == 5:
         prompts = prompt_EP_5

    context_list = []
    context_list.append({
      'role': 'system',
      'content': prompts
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


def main():

    fw = open(config.save_path, mode="a")
    data_test = np.load("data/ED/sys_dialog_texts.test.npy", allow_pickle=True)
    len_test = len(data_test)
    continueid = -1     # continue after interruption, default to -1
    id = continueid + 1
    while id < len_test:
        # try:
        #     res = get_conv(config.model, data_test[id], config.few_shot)
        # except:
        #     continue
        res = get_conv(config.model, data_test[id], config.few_shot)
        fw.write(str(id) + " #*#*# " + res + "\n")
        print(id)
        id = id + 1
        # time.sleep(20)
    fw.close()

if __name__ == '__main__':
    main()


