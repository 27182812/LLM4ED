import json
import openai
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
import random

import config

random.seed(13)

### initial prompt

prompt_EP_0 = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
            Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is the existing dialogue context:\n\n"
prompt_EP_1_origin = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
        Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is an instance:\n\n"
prompt_EP_5_origin = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
        Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener.\n\n" + "The following is some instances:\n\n"

data_train_dialog = np.load("data/ED/sys_dialog_texts.train.npy", allow_pickle=True)
data_train_target = np.load("data/ED/sys_target_texts.train.npy", allow_pickle=True)

len_train = len(data_train_dialog)
shot1 = random.randint(0, len_train - 1)
shot5 = random.sample(range(0, len_train), 5)

tmp = data_train_dialog[shot1]
tmp.append(data_train_target[shot1].strip())

Instance = "Instance:\n"
for i, j in enumerate(tmp):
    if i % 2:
        Instance += "Speaker: " + j + "\n"
    else:
        Instance += "Listener: " + j + "\n"

prompt_EP_1_origin = prompt_EP_1_origin + Instance

Instances = ""
for id, k in enumerate(shot5):
    tmp = data_train_dialog[k]
    tmp.append(data_train_target[k].strip())
    Instances += "Instance " + str(id + 1) + ":\n"
    for i, j in enumerate(tmp):
        if i % 2:
            Instances += "Speaker:" + j + "\n"
        else:
            Instances += "Listener: " + j + "\n"
prompt_EP_5_origin = prompt_EP_5_origin + Instances

### your openai.api_key
openai.api_key = config.apikey

def query_openai_complete(query, engine="gpt-35-turbo"):
    if engine == 'chatgpt':
        engine = "gpt-35-turbo"
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=query,
            # max_tokens=2000,
            temperature=config.temperature,
            stop=["<END>"])
    except TypeError as e:
        print(e)
        return {"choices": [{"text": ""}]}
    # return response
    return response["choices"][0]["text"]


def main():

    fw = open(config.save_path, mode="a")
    data_test = np.load("data/ED/sys_dialog_texts.test.npy", allow_pickle=True)
    len_test = len(data_test)
    continueid = -1     # continue after interruption, default to -1
    id = continueid + 1
    while id < len_test:
        prompts = ""
        for j, text in enumerate(data_test[id]):
            if len(text) == 0:
                continue
            if j % 2 == 0:
                role_str = 'Speaker: '
            else:
                role_str = 'Listener: '

            prompts += role_str + text + '\n'
        prompts += "Listener:"
        prompt_EP_0 = prompt_EP_0 + prompts
        prompt_EP_1 = prompt_EP_1_origin + "\nThe following is the existing context:\n\n" + prompts
        prompt_EP_5 = prompt_EP_5_origin + "\nThe following is the existing context:\n\n" + prompts
        # try:
        #     res = query_openai_complete(prompt_EP_5, config.model) # "text-davinci-003, "davinci"
        # except:
        #     continue
        
        res = query_openai_complete(prompt_EP_5, config.model) # "text-davinci-003, "davinci"
        
        fw.write(str(id) + " #*#*# " + res + "\n")
        print(id)
        id = id + 1
        # time.sleep(20)
    fw.close()

if __name__ == "__main__":
    main()


