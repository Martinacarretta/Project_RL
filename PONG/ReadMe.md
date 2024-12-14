## PROJECT - PONG
This repository contains the files used to train agent to play [Pong](FALTA LINK)

## The game:
You control the right paddle, you compete against the left paddle controlled by the computer. You each try to keep deflecting the ball away from your goal and into your opponent’s goal.
FALTA MIRAR SI ES AQUESTA VERSIO DE PONG O LA DE GYM

### Action space:
Pong's action space is Discrete(6). The different actions correspond to:
- 0: NOOP
- 1: FIRE
- 2: RIGHT
- 3: LEFT
- 4: RIGHTFIRE
- 5: LEFTFIRE

## Observation space:
Atari environments have three possible observation types: "rgb", "grayscale" and "ram".

- obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)
- obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)
- obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8), a grayscale version of the “rgb” type


## Untrained agent:
First and foremost, we wanted to see how an untrained agent would perform to have an ide on how the game worked.
The gifs below represents an untrained agent. 

FALA POSAR GIFS (3) DEL UNTRAINED PONG

FALTA POSAR EL PLOT DELS REWARDS DEL TEST DE 10 EPISODIS UNTRAINED

## Training:
We decided to tackle this part of the project with:
- FALTA

## Training - FALTA:
This part of the training has FALTA sections:

### Requirements:
The code requires the following libraries:
FALTA

to install them all, you can use:
FALTA

### Preprocessing:

FALTA

### Training:

FALTA parlar una mica del process que hem seguit

The training process is as follows:

FALTA

### Results:
The model is under the name "FALTA .pth"
FALTA GIFS DEL TEST DEL DQN I EL PLOT DELS REWARDS

### Conclusion:
FALTA