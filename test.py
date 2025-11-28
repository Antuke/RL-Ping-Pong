'''
Task:   Load the trained model and play the game
Group : 19
Member:
        Antonio Sessa 0622702305 a.sessa108@studenti.unisa.it 
        Angelo Molinario 0622702311 a.molinario3@studenti.unisa.it
        Massimiliano Ranauro 0622702373 m.ranauro2@studenti.unisa.it
        Pietro Martano 0622702402 p.martano@studenti.unisa.it
'''

from utils.nets import InverseKinematicsModel
from client import Client
from ddpg import DDPG
import numpy as np
import torch
import math
from math import sqrt
import sys
import time

GAMMA = 0.15
TAU = 0.0001
HIDDEN_SIZE = (128, 256, 256)
NUM_INPUT  = 5
ACTION_SPACE =np.random.rand(1)
CHECKPOINT_DIR = "./checkpoint/"
BATCH_SIZE = 64
REPLAY_SIZE = 10000
SIGMA = 0.015

def expected_ball_drop_point(x0, y0, z0, vel_x, vel_y, vel_z, height=0.4, verbose=False):
    '''
    Compute the expected ball's drop point at a given z.        
      x(t)= vel_x*t + x0
      y(t)= vel_y*t + y0
      z(t)= -1/2*g*t^2 + vel_z*t + z0
    Args:
        x0:       Ball's initial x position 
        y0:       Ball's initial y position 
        z0:       Ball's initial z position 
        vel_x:    Ball's initial x velocity 
        vel_y:    Ball's initial y velocity
        vel_z:    Ball's initial z velocity
        heigth:   Heigth of interest        
        verbose:  Debug information
    Return:
        (x,y,z)   ball's drop point to a given z if exists, None otherwise
    '''
    g = 9.81
    delta= (vel_z)**2 - 4*(-0.5*g)*(z0-height)

    if( delta<0 ):
      return None

    time_add_positive= ( -(vel_z) + sqrt( delta ) ) / ( 2*(-0.5*g) )
    time_add_negative= ( -(vel_z) - sqrt( delta ) ) / ( 2*(-0.5*g) )

    time= max(time_add_positive, time_add_negative)
    if time < 0:
        return None
    if verbose:
      print(f"Time used: {time} sec")

    return ( float(vel_x*time+x0) , float(vel_y*time+y0) , float(height) )

def get_angle_normale(v1, v2):
    '''
        Comput the angle between two vectors
        Args:
            normale : Vettore contenente le componenti (x,y,z) della normale alla racchetta
            v1: Vettore contenente le componenti (x,y,z) rispetto al quale calcolare l'angolo
        Return:
            theta_d : Angolo in gradi tra i due vettori
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)

    norm_V1 = np.linalg.norm(v1)
    norm_V2 = np.linalg.norm(v2)

    dot_prod = np.dot(v1, v2)
    if norm_V1==0 or norm_V2==0:
        return -1
    cos_theta = dot_prod / (norm_V1 * norm_V2)
    theta_r = np.arccos(cos_theta)     

    return theta_r


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


def run(cli, model, agent):
    jp = [0.0] * 11
    jp[10] = 1.57
    print("[+] connected to server, let's play!\n--------------------")
    flag_shot = False
    state = cli.get_state()    
    with torch.no_grad():
        while True:
            if state[28]:
                # The ball hit my half side but not my robot
                    if state[30] == 1 and state[31] == 0:
                        pred = expected_ball_drop_point(state[17], state[18], state[19], state[20], state[21],
                                                            state[22], height=0.2)  
                        if pred is not None:
                            if pred[1] < -0.8:
                                alpha = 0.14
                            else:
                                alpha = 0.1
                            h = 0.2 + alpha*(-pred[1])                        
                            pred = expected_ball_drop_point(state[17], state[18], state[19], state[20], state[21],
                                                                state[22], height=h)                    
                                            
                        if pred is not None:
                            forward = torch.tensor(pred, dtype=torch.float32)
                            joints = model(forward)
                            jp = update_jp(jp, joints)
                            cli.send_joints(jp)

                        b = np.array(state[17:20])
                        t = np.array(state[11:14])
                        distance = np.linalg.norm(b - t)
                        
                        if distance < 0.3 and flag_shot == 0:                        
                            # ball velocity x,y,z
                            vb = np.array(state[20:23])

                            # Angle between the paddle's normale and the ball velocity                        
                            angolo = get_angle_normale(state[14:17],state[20:23])
                            # magnitudien velocity
                            velocity = math.sqrt(vb[0]**2 + vb[1]**2 + vb[2]**2)

                            # ball's depth and height
                            by = state[18]
                            bz = state[19]      
                            st = torch.tensor([angolo, velocity, by, bz, distance],dtype=torch.float32)  
                            print(f'stato palla = velocità = {velocity}, angolo={angolo}, by={by}, distance={distance}')                        
                            action = agent.calc_action(st, None).cpu()                              
                            
                            if state[11] > 0.48:
                                roll = 1.57 - 0.1
                            elif state[11] < -0.48:
                                roll = 1.57 + 0.1
                            else:
                                roll = 1.57
                            
                            jp[9] += action[0]   
                            jp[10] = roll
                            cli.send_joints(jp)
                            jp[9] -= action[0]                            
                            time.sleep(0.05)                                                                        
                            cli.send_joints(jp)
                            flag_shot = 1                                                                                                                                   
                    elif state[30] == 0:
                        pred = expected_ball_drop_point(state[17], state[18], state[19], state[20], state[21],
                                                                state[22], height=0.1)
                        jp[1] = state[17]                                                
                        if pred is not None:
                            if pred[1]<-0.1: 
                                if pred[0] > 0:
                                    jp[1] = -0.4
                                else:
                                    jp[1] = 0.4                                                               
                        jp[3] = 0.3
                        jp[5] = -1.3
                        jp[7] = -1.3
                        jp[9]= -0.3
                        jp[10] = 1.57
                        cli.send_joints(jp)
                                        
                    if flag_shot and state[32]:
                        flag_shot = False                    

            state = cli.get_state()


def main():
    name = 'Group 19'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    host = 'localhost'
    if len(sys.argv) > 2:
        host = sys.argv[2]

    print('[+] loading inverse kinematic model...')
    model = InverseKinematicsModel()
    model.load_state_dict(torch.load('./models/6j_256.pth', map_location=torch.device('cpu')))
    model.eval()    
        
    agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE,
                 NUM_INPUT,
                 ACTION_SPACE,
                 checkpoint_dir=CHECKPOINT_DIR
                 )
    agent.load_checkpoint("./models/agent.pth.tar")
    agent.calc_action(torch.tensor([0,0,0,0,0], dtype=torch.float32) ,None)
    print("[+] Done loading agent")
    print('[+] connecting to server')
    cli = Client(name, host)
    run(cli, model, agent)


if __name__ == '__main__':
    main()
