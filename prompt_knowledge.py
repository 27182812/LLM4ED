### ED

import os
import openai
import numpy as np
import time
import nltk
import torch

import config
from comet import Comet

WORD_PAIRS = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}

### 任务说明（Instruction）+对话语境

reldef = '''xIntent represents their intent before the event.
xNeed represents what they need in order for the event to happen.
xWant represents what they would want after the event.
xEffect represents the effect of the event on the person.
xReact represents their reaction to the event.
'''
relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
rels = ["x_intent:", "x_need:", "x_want:", "x_effect:", "x_react:"]

knowledge_origin = "Don't rush to reply, I can provide the following additional knowledge to help you provide a better reply. The following are the definitions of the five commonsense relations, followed by the content of the five relations extracted from the existing conversation. You can combine them and the dialogue context generates the final reply."

### your openai.api_key
openai.api_key = config.apikey

def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in WORD_PAIRS.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

def get_commonsense(comet, sen, others=None):
    cs_list = []
    input_event = " ".join(sen)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append([" ".join(i) for i in cs_res])
    # print(cs_list)
    return cs_list


def get_res_chatgpt(m_name, contents):
    response = openai.ChatCompletion.create(
      model=m_name, 
      messages=contents, 
      temperature=config.temperature,
    )['choices'][0]['message']['content']

    return response

def get_conv(m_name, context):

    context_list = []
    context_list.append({
      'role': 'system',
      'content': "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
  Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener."
    })

    for i, text in enumerate(context[:-1]):
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

    add_info = knowledge_origin + '\n' + reldef

    sen = process_sent(context[-1])
    comet = Comet("data/Comet", config.device)
    others = get_commonsense(comet, sen)

    for rel, con in zip(rels, others):
        add_info += rel + con + '\n'
    # print(add_info)

    context_list.append({
        'role': 'user',
        'content': context[-1] + '\n\n' + add_info
    })

    response = get_res_chatgpt(m_name, context_list)

    return response

def main():

    fw = open(config.save_path, mode="a")
    # rels_con = np.load("results/comet.npy", allow_pickle=True)

    data_test = np.load("data/ED/sys_dialog_texts.test.npy", allow_pickle=True)
    len_test = len(data_test)
    continueid = -1     # continue after interruption, default to -1
    id = continueid + 1
    while id < len_test:
        # try:
        #     res = get_conv(config.model, data_test[id])
        #     # res = get_conv(config.model, data_test[id], rels_con[id])
        # except:
        #     continue
        
        res = get_conv(config.model, data_test[id])
        
        fw.write(str(id) + " #*#*# " + res + "\n")
        print(id)
        id = id + 1
        # time.sleep(20)
    fw.close()

if __name__ == '__main__':
    main()