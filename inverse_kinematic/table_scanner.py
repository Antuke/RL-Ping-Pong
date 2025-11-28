'''
Task:	Scan the table to create an inverse kinematics dataset
Group:	19
Member:
	Antonio Sessa 0622702305 a.sessa108@studenti.unisa.it	
	Angelo Molinario 0622702311 a.molinario3@studenti.unisa.it
	Massimiliano Ranauro 0622702373 m.ranauro2@studenti.unisa.it
	Pietro Martano 0622702402 p.martano@studenti.unisa.it

'''


from client import Client, JOINTS
import sys
import math
import numpy as np
import time
import csv

pi = math.pi


def stance_1(x, y):
    jp = [0.0] * 11
    a = -math.pi / 3.8
    jp[0] = x
    jp[1] = y
    jp[5] = a
    jp[7] = a
    jp[9] = -math.pi / 3.5
    jp[10] = math.pi / 2
    return jp


def stance_2(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -pi / 10
    j[7] = -pi * 3 / 4
    j[5] = 0
    j[10] = math.pi / 2
    return j


def stance_3(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.9
    j[5] = -0.6
    j[7] = -0.3

    j[10] = math.pi / 2
    return j


def stance_4(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.1
    j[5] = -1
    j[7] = 0
    j[9] = -1.4

    j[10] = math.pi / 2
    return j


def stance_5(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.5
    j[5] = -1
    j[7] = 0
    j[9] = -1.0

    j[10] = math.pi / 2
    return j


# KEEP
def stance_6(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.9
    j[5] = -0.6
    j[7] = 0.1
    j[9] = -0.6
    j[10] = math.pi / 2
    return j


def stance_7(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = 0
    j[5] = 0
    j[7] = -1
    j[9] = -1.8
    j[10] = math.pi / 2
    return j


def stance_8(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.5
    j[5] = 0
    j[7] = 0
    j[9] = -2
    j[10] = math.pi / 2
    return j


def stance_9(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.5
    j[5] = 0
    j[7] = -1
    j[9] = -1.4
    j[10] = math.pi / 2
    return j


def stance_10(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = 0
    j[5] = -0.6
    j[7] = -0.6
    j[9] = -1.6

    j[10] = math.pi / 2
    return j


def stance_11(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -3.0 / 2
    j[5] = 0
    j[7] = 0
    j[9] = -0.15

    j[10] = math.pi / 2
    return j


def stance_12(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -1.5 / 2
    j[5] = 0
    j[7] = 0
    j[9] = -0.15

    j[10] = math.pi / 2
    return j


def stance_14(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.3
    j[5] = -0.49
    j[7] = -2
    j[9] = -0.3
    j[10] = math.pi / 2

    return j


def stance_13(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.3
    j[5] = -0.5
    j[7] = -2.1
    j[9] = -0.1
    j[10] = math.pi / 2

    return j


def stance_15(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.3
    j[5] = -0.19
    j[7] = -2.4
    j[9] = -0.1
    j[10] = math.pi / 2

    return j


def stance_16(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.1
    j[5] = -0
    j[7] = -2.4
    j[9] = -0.1
    j[10] = math.pi / 2

    return j


def stance_17(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -0.1
    j[5] = -0
    j[7] = -2.4
    j[9] = -0.1
    j[10] = math.pi / 2

    return j


def stance_18(x, y):
    j = [0.0] * 11
    j[0] = x
    j[1] = y
    j[3] = -pi / 10
    j[7] = -2.356
    j[5] = -0.1
    j[10] = math.pi / 2
    return j


def reset():
    j = [0.0] * 11
    return j


step1 = 0.01
step2 = 0.01


def run(cli, stance, header=False):
    state = cli.get_state()
    j = [0.0] * 11
    cli.send_joints(stance(state[0],state[1]))

    np.set_printoptions(suppress=True, precision=4)
    time.sleep(0.1)
    # Open a CSV file to write the dataset
    with open('dataset_nuovo_6joint.csv', 'a', newline='') as csvfile:
        dataset_writer = csv.writer(csvfile)

        if header == True:
            dataset_writer.writerow(
                ["JOINT0", "JOINT1", "JOINT3", "JOINT5", "JOINT7", "JOINT9", "TARGET_X", "TARGET_Y", "TARGET_Z", 'NX',
                 'NY', 'NZ'])

        for i in range(int(-0.3 / step1), int(0.3 / step1) + 1):
            value1 = i * step1
            print(f'value1 = {value1}')
            j[0] = value1
            for k in range(int(-0.79 / step2), int(0.79 / step2) + 1):
                value2 = k * step2
                j[1] = value2

                j = stance(value1, value2)
                cli.send_joints(j)
                time.sleep(0.060)
                state = cli.get_state()
                dataset_writer.writerow(
                    [f"{state[0]:.4f}", f"{state[1]:.4f}", f"{state[3]:.4f}", f"{state[5]:.4f}", f"{state[7]:.4f}",
                     f"{state[9]:.4f}", f"{state[11]:.4f}", f"{state[12]:.4f}", f"{state[13]:.4f}", f"{state[14]:.4f}",
                     f"{state[15]:.4f}", f"{state[16]:.4f}"])


def main():
    name = 'Client Scanner'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    host = 'localhost'
    if len(sys.argv) > 2:
        host = sys.argv[2]

    cli = Client(name, host,port=29)
    stances = [
        stance_18, stance_15, stance_14, stance_13, stance_17, stance_16, stance_1, stance_2, stance_4, stance_5,
        stance_9, stance_10
    ]
    i = 1
    for stance in stances:
        print(f'starting stance {i}')
        if i == 1:
            run(cli, stance, True)
        else:
            run(cli, stance)
        i = i + 1


if __name__ == '__main__':
    main()
