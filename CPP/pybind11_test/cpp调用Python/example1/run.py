import os
from time import sleep

with open('input.txt','w') as f:
    f.write("lalala")
    f.flush()
    f.close()
print("end writing...")

sleep(1)

