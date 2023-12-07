### ED


### CoT
CoT = "Don't rush to reply yet, let's think step by step. Based on the above conversation, what may be the user's emotion (One of these 32 emotions (guilty, caring, lonely, excited, sad, hopeful, angry, joyful, disappointed, faithful, content, annoyed, terrified, nostalgic, grateful, trusting, surprised, ashamed, impressed, proud, furious, sentimental, confident, anxious, jealous, apprehensive, embarrassed, anticipating, disgusted, devastated, prepared, afraid).), and according to his description, what may be the situation when he feels this way?"
# add_info = "I think based on the above conversation, it is predicted that the user's emotion may be "
# add_info = "I think based on the above conversation, it is predicted that the user's situation may be "

import os
import openai
import numpy as np
import time

import config

### your openai.api_key
openai.api_key = config.apikey

def get_res_chatgpt(m_name, contents):
    response = openai.ChatCompletion.create(
      model=m_name, 
      messages=contents, 
      temperature=config.temperature,
    )['choices'][0]['message']['content']

    return response

def get_conv(m_name, context, stage, CoT_con=None):
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
    context_list.append({
        'role': 'user',
        'content': context[-1] + "\n\n" + CoT
    })
    
    if stage == 2:
        context_list.append({
            'role': 'assistant',
            'content': CoT_con
        })
        context_list.append({
            'role': 'user',
            'content': "Now combine your thoughts with the existing dialogue context and give your response."
        })
    
    response = get_res_chatgpt(m_name, context_list)

    return response

def main():

    if config.stage == 2:
        CoTs = open("results/ED_chatgpt_stage1.txt", mode="r").readlines()
        fw = open(config.save_path, mode="a")
    elif config.stage == 1:
        fw = open("results/ED_chatgpt_stage1.txt", mode="a")
    data_test = np.load("data/ED/sys_dialog_texts.test.npy", allow_pickle=True)
    len_test = len(data_test)
    continueid = -1     # continue after interruption, default to -1
    id = continueid + 1
    while id < len_test:
        # try:
        #     if config.stage == 2:
        #         res = get_conv(config.model, data_test[id], 2, CoTs[id])
        #     elif config.stage == 1:
        #         res = get_conv(config.model, data_test[id], 1)
        # except:
        #     continue
       
        if config.stage == 2:
            res = get_conv(config.model, data_test[id], 2, CoTs[id])
        elif config.stage == 1:
            res = get_conv(config.model, data_test[id], 1)
       
        fw.write(str(id) + " #*#*# " + res + "\n")
        print(id)
        id = id + 1
    fw.close()

if __name__ == '__main__':
    main()

