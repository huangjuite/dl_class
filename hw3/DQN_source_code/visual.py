# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import matplotlib.pyplot as plt
import re

filepath = 'log3.txt'
log = []
eval_log = []
eval_epi = []
epi = 0
with open(filepath) as fp:
    while True:
        line = fp.readline()
        if line is '':
            break
        rt = re.findall('reward: +[\d]+',line)
        
        if len(rt)==2:
            r = re.findall('[\d]+',rt[0])
            eval_log.append(int(r[0]))
            eval_epi.append(epi)
            
        r = re.findall('[\d]+',rt[-1])
        log.append(int(r[0]))
        epi+=1
        
plt.figure(figsize=(20,6))
plt.title('learning DQN episodes total reward')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.plot(log)
plt.savefig('epi_reward')
plt.close()

plt.figure(figsize=(20,6))
plt.title('eval DQN episodes total reward')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.plot(eval_epi,eval_log)
plt.savefig('epi_eval')
plt.close()
