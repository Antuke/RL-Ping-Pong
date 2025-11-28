import logging
import numpy as np
from client import Client
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
import torch
import math
from utils import Trajectory
'''
Task:   Use the trained model to play the game.
Group : 19
Member:
        Antonio Sessa 0622702305 a.sessa108@studenti.unisa.it 
        Angelo Molinario 0622702311 a.molinario3@studenti.unisa.it
        Massimiliano Ranauro & 0622702373 m.ranauro2@studenti.unisa.it
        Pietro Martano 0622702402 p.martano@studenti.unisa.it
'''

import sys
from utils.nets import InverseKinematicsModel
from ddpg import DDPG
import time
from utils.nets import InverseKinematicsModel

from torch.utils.tensorboard import SummaryWriter


GAMMA = 0.15
TAU = 0.0001
HIDDEN_SIZE = (128, 256, 256)
NUM_INPUT  = 5
ACTION_SPACE =np.random.rand(1)
CHECKPOINT_DIR = "./checkpoint_service_last/"
BATCH_SIZE = 64
REPLAY_SIZE = 10000
SIGMA = 0.01

# Monitoraggio del training per la reward media con tensorBoard
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))
writer = SummaryWriter('runs/service_last2')    


def get_angle_normale(normale, v1):
    '''
        Funzione che calcola l'angolo tra la normale e un vettore.
        Args:
            normale : Vettore contenente le componenti (x,y,z) della normale alla racchetta
            v1: Vettore contenente le componenti (x,y,z) rispetto al quale calcolare l'angolo
        Return:
            theta_d : Angolo in gradi tra i due vettori
    '''
    normale = np.array(normale)
    v1 = np.array(v1)
    
    norm_N = np.linalg.norm(normale)
    norm_V =  np.linalg.norm(v1)
    
    if norm_N==0 or norm_V==0:
        return -1
    
    dot_prod = np.dot(normale,v1)
    
    cos_theta = dot_prod/(norm_N*norm_V)
    theta_r = np.arccos(cos_theta) #angolo in radianti
 
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


def is_final_episode(old_score, new_score):
    '''
        Args:
            Old_score : (my_core, opponent_score)
            new_score : (my_core, opponent_score)
        Returns:
            True if one of the player's scores, false otherwise
    '''    
    if old_score[0] != new_score[0]:    
        return True
    if old_score[1] != new_score[1]:
        return True
    
    return False


def get_score_reward(score):
    score_reward = 0
    if score[0][0] < score[1][0]:
        score_reward = 10
    if score[0] [1] < score[1][1]:
        score_reward = -10
    return score_reward

def get_trajectory_reward(y, verbose=False):                 
    if y<1:
        trajectory_reward = -(1-y**2)
    elif 1<= y <=1.4:
        trajectory_reward = 3
    elif 1.4< y<=1.8:
        trajectory_reward = 5
    elif 1.8< y <=2.18:
            trajectory_reward = 4       
    else:                
        trajectory_reward = - abs(y - 1)
    if verbose:
        print(f"Pred {y} - reward {trajectory_reward}")            
    return float(trajectory_reward)


def run(cli, model, agent, memory, noise, start_ep=0):
    numero_episodi = 20000
    ep_number = start_ep
    jp = [0.0] * 11
    jp[10] = math.pi / 2
    prev_reward = -999
    print("[+] connected to server, let's play!\n--------------------")
    score = [(0,0),(0,0)]    # [old_score(my_score, opponent_score), new_score(my_score, opponent_score)]
    action = None            
    episode_state_list = {}    
    while ep_number<numero_episodi:
        noise.reset()
        state = cli.get_state()
        score[0] = score[1] # Aggiorno il vecchio score con il nuovo
        done = False # L'episodio è terminato
        flag_shot = 0
        eps_reward = 0
        
        print('----------------------------------')
        print("New epidose")
        traj_reward = None
        
        index_state_list = 0        
        episode_state_list.clear()
        # Start single episode
        while True:
            score[1] = (state[34], state[35])
            if state[28]:
            # la palla ha colpito il tavolo e non ha ancora colpito il robot                
                if state[30] == 1 and state[31] == 0:
                    pred = Trajectory.expected_ball_drop_point(state[17], state[18], state[19], state[20], state[21],
                                                            state[22], height=0.2)  
                    if pred is not None:
                        if pred[1] < -0.8:
                            alpha = 0.14
                        else:
                            alpha = 0.1
                            
                        h = 0.2 + alpha*(-pred[1])                        
                        pred = Trajectory.expected_ball_drop_point(state[17], state[18], state[19], state[20], state[21],
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

                        # angolo normale                        
                        angolo = get_angle_normale(state[14:17],state[20:23])
                        # magnitudien velocity
                        velocity = math.sqrt(vb[0]**2 + vb[1]**2 + vb[2]**2)

                        # profondità palla
                        by = state[18]
                        bz = state[19]
                        st = torch.tensor([angolo, velocity, by, bz, distance],dtype=torch.float32)                        
                        #print(f'stato palla = velocità = {velocity}, angolo={angolo}, by={by}, distance={distance}')                        
                        action = agent.calc_action(st, noise).cpu()  
                        episode_state_list[index_state_list]  = [st, action, 0 , torch.tensor([0,0,0,0,0], dtype=torch.float32).to(device)]
                        
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
                    pred = Trajectory.expected_ball_drop_point(state[17], state[18], state[19], state[20], state[21],
                                                            state[22], height=0.1)
                    jp[1] = state[17]
                    if pred is not None:
                        if pred[1]<-0.1:
                            jp[1] = -5*state[17]                    
                        
                    jp[3] = 0.3
                    jp[5] = -1.3
                    jp[7] = -1.3
                    jp[9]= -0.3
                    jp[10] = 1.57
                    cli.send_joints(jp)
                     
            # Controllo dove cade la pallina dopo che ho colpito la pallina                
                op_ball_dist = np.linalg.norm(state[23:26]-state[17:20])
                if (0.0 <= state[19] <= 0.1 or op_ball_dist<0.2) and (flag_shot and not state[33]):
                    flag_shot = False
                    traj_reward = get_trajectory_reward(state[18])    
                    print(f"Altezza della pallina {state[19]} - op_dist = {op_ball_dist}") 
                    episode_state_list[index_state_list][2] = traj_reward                    
                    if index_state_list >= 1:
                        episode_state_list[index_state_list-1][3] = episode_state_list[index_state_list][0] # Aggiorno il next state
                    index_state_list += 1                   
                
            if is_final_episode(score[0], score[1]) and index_state_list !=0:             
                # Se l'indice è 0 vuol dire che non è stata registrata la reward per il colpo
                                        
                index_state_list -= 1 # L'utlimo indice non è valido perché ha informazioni non valide
                print(f"Final state (transizioni osservate : {len(episode_state_list)})")           
                score_reward = get_score_reward(score)                
                episode_state_list[index_state_list][2]+=score_reward 
                ep_reward  = 0
                # inserisco le transizioni registrate nel replay buffer
                for i in range(len(episode_state_list)):
                    ep_reward += episode_state_list[i][2]
                    
                    state =  episode_state_list[i][0]
                    action = episode_state_list[i][1]
                    reward = torch.tensor([episode_state_list[i][2]], dtype=torch.float32).to(device)
                    done = torch.tensor([i==index_state_list], dtype=torch.float32).to(device)
                    next_state = episode_state_list[i][3].to(device)
                    print(f"[+] episode {ep_number}")
                    print(f"\tst {state.detach().to('cpu').numpy()},\n\t action {action.detach().to('cpu').numpy()},\n\t reward {reward.detach().to('cpu').numpy()},\n\t n_st {next_state.detach().to('cpu').numpy()},\n\t done {done.detach().to('cpu').numpy()}")
                    memory.push(state, action, reward, done, next_state) 
                writer.add_scalar('train/mean_reward_for_ep',ep_reward/len(episode_state_list) , ep_number)  
                eps_reward += ep_reward /len(episode_state_list)
                ep_number += 1                   
                break
            if is_final_episode(score[0], score[1]) and index_state_list ==0: 
                print("Transizioni perse")                
                break
                                
            state = cli.get_state()                        
                                                
            
        if ep_number % 32 == 0 and ep_number != 0:
                                            
            mean_reward = eps_reward / 32
            eps_reward = 0
            writer.add_scalar('train/mean_reward_for_32_ep',mean_reward , ep_number)  
            if mean_reward >= prev_reward:
                agent.save_checkpoint(ep_number)                
                prev_reward = mean_reward                        
            print(f'Reward medio = {mean_reward}')                                   
            epoch_policy_loss = 0
            epoch_value_loss = 0
            if (len(memory)> BATCH_SIZE):         
                print(f"Optimizing ...")                             
                for i in range(32):
                    transitions = memory.sample(BATCH_SIZE)                
                    batch = Transition(*zip(*transitions))                
                    value_loss, policy_loss = agent.update_params(batch)                
                    epoch_value_loss += value_loss
                    epoch_policy_loss += policy_loss
                writer.add_scalar('train/value_loss',epoch_value_loss/32, ep_number)
                writer.add_scalar('train/policy_loss',epoch_policy_loss/32, ep_number)    
        if((ep_number+1) % 101 == 0):
            agent.save_checkpoint(ep_number) 

def main():
    name = 'TRAIN_Q_TARGET'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    host = 'localhost'
    if len(sys.argv) > 2:
        host = sys.argv[2]

    print('[+] loading inverse kinematic model...')
    model = InverseKinematicsModel()
    model.load_state_dict(torch.load('.\\models\\6j_256.pth', map_location=torch.device('cpu')))
    model.eval()
    
    
    nb_actions = ACTION_SPACE.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(SIGMA) * np.array(nb_actions))
    
    
    agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE,
                 NUM_INPUT,
                 ACTION_SPACE,
                 checkpoint_dir=CHECKPOINT_DIR
                 )        
    start_ep = agent.load_checkpoint()
    agent.calc_action(torch.tensor([0,0,0,0,0], dtype=torch.float32) ,None)
    
    print("[+] Done creating agent")
    memory = ReplayMemory(int(REPLAY_SIZE))
    print("[+] Done creating the replay buffer")
    print('[+] connecting to server')    
    cli = Client(name, host,port=333)
    run(cli, model, agent, memory, ou_noise, 0)


if __name__ == '__main__':
    main()
