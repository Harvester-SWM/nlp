import sys
import os

#print(sys.path)

#root_path = sys.path[0][:-6]
root_path = sys.path[0]
model_path = os.path.join(root_path, '../models')

#print(model_path)
sys.path.append(model_path)

print(sys.path)

from test_model import model_mk1

model_mk1()

from new_model import model_mk1

model_mk1()