import pandas as pd
import riiideducation
import draw_tools.draw as draw
import numpy as np
import collections

env = riiideducation.make_env()
dtype = {'row_id': 'int64',
         'timestamp': 'int64',
         'user_id': 'int32',
         'content_id': 'int16',
         'content_type_id': 'int8',
         'task_container_id': 'int16',
         'user_answer': 'int8',
         'answered_correctly': 'int8',
         'prior_question_elapsed_time': 'float32',
         'prior_question_had_explanation': 'boolean'}
train_df = pd.read_csv('train.csv', low_memory=False, dtype=dtype)
train_df = train_df.to_dict(orient='records')
user_dict = dict()
for item in train_df:
    user_id = item['user_id']
    temp_dict = user_dict.get(user_id, dict())
    if item['content_id'] not in temp_dict:
        temp_dict[item['content_id']] = item['answered_correctly']
        user_dict[user_id] = temp_dict

question_dict = dict()

for user in user_dict:
    for question in user_dict[user]:
        correctness = user_dict[user][question]
        temp_list = question_dict.get(question, list())
        temp_list.append(correctness)
        question_dict[question] = temp_list

correct_rate = dict()
rate_list = list()
number_list = list()
for question in question_dict:
    correct_rate[question] = np.average(question_dict[question])
    rate_list.append(correct_rate[question])
    number_list.append(len(question_dict[question]))

draw.one_d_dot_graph(rate_list, 0, 1)
c = collections.Counter(number_list)
keys = c.keys()
values = list()
for k in keys:
    values.append(c[k])
draw.dis_graph(keys, values)