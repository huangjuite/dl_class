import random
import numpy as np

t0 = 0
t1 = 0
t2 = 0

# print([np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])])
# print(random.sample([0, 1, 2], 1))
for i in range(1000):
    t = random.sample([0,0,0,1,1,1,1,1,1,2], 1)[0]
    if t == 0:
        t0 += 1
    elif t==1:
        t1 += 1
    elif t==2:
        t2 += 1
    
print(t0,t1,t2)
    
