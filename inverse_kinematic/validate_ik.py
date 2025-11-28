'''
Task:	Scan the table to create an inverse kinematics dataset
Group:	19
Member:
	Antonio Sessa 0622702305 a.sessa108@studenti.unisa.it	
	Angelo Molinario 0622702311 a.molinario3@studenti.unisa.it
	Massimiliano Ranauro 0622702373 m.ranauro2@studenti.unisa.it
	Pietro Martano 0622702402 p.martano@studenti.unisa.it

'''
import math
import time

from client import Client, JOINTS
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd



def update_jp(jp, j):
    jp[0] = j[0]
    jp[1] = j[1]
    jp[2] = 0
    jp[3] = j[2]
    jp[5] = j[3]
    jp[7] = j[4]
    jp[9] = j[5]
    jp[10] = math.pi / 2
    return jp

class InverseKinematicsModel(nn.Module):
    def __init__(self):
        super(InverseKinematicsModel, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x



def run(cli, model):
    input_csv = 'dataset_nuovo_6joint.csv'
    data = pd.read_csv(input_csv)
    jp = [0.0] * 11
    jp[10] = math.pi / 2
    total_distance = 0
    target_columns = data[['TARGET_X', 'TARGET_Y', 'TARGET_Z']].sample(500)

    inputs = target_columns.values.astype(np.float32)

    for i in range(len(inputs)):
        input_value = inputs[i].reshape(1, -1)[0]  # Reshape the input to (1, 3)
        input_value[0] += np.random.uniform(-0.03, 0.03)
        input_value[1] += np.random.uniform(-0.03, 0.03)
        input_value[2] -= np.random.uniform(-0.03, 0.03)
        forward = torch.tensor(input_value,dtype=torch.float32)
        joints = model(forward)
        jp = update_jp(jp, joints)
        cli.send_joints(jp)
        time.sleep(2.0)
        state = cli.get_state()

        actual_pos = state[11:14]
        distance = np.linalg.norm(actual_pos - input_value)
        total_distance += distance
        print(f'actual pos = {actual_pos}, desired_pos = {input_value},distance = {distance}')
        time.sleep(2.0)

    mean_distance = total_distance / len(inputs)

    print(f'Mean distance between actual_pos and input_value: {mean_distance}')

def main():
    name = 'Client Training1'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    host = 'localhost'
    if len(sys.argv) > 2:
        host = sys.argv[2]

    print('[+] loading inverse kinematic model...')
    model = InverseKinematicsModel()
    model.load_state_dict(torch.load('models/6joint_256_new.pth', map_location=torch.device('cpu')))
    model.eval()
    print('[+] connecting to server')
    cli = Client(name, host)
    run(cli, model)


if __name__ == '__main__':
    main()
