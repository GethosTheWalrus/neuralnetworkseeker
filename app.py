import pygame
from keras.models import Sequential
from keras.layers.core import Activation, Dense
import os
import random
import math
import numpy as np

clock = pygame.time.Clock()

resolution_x = 600
resolution_y = 600

rect_dim = {
    "rect_x": 290,
    "rect_y": 290,
    "rect_w": 10,
    "rect_h": 10,
}

foods = {

    "food_dim1": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10,
        "r": 255,
        "g": 128,
        "b": 128
    },
    "food_dim2": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10,
        "r": 255,
        "g": 128,
        "b": 128
    },
    "food_dim3": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10,
        "r": 255,
        "g": 128,
        "b": 128
    },
    "food_dim4": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10,
        "r": 255,
        "g": 128,
        "b": 128
    },
    "food_dim5": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10,
        "r": 255,
        "g": 128,
        "b": 128
    }

}

model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(128, input_dim=128, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

pygame.init()
screen = pygame.display.set_mode((resolution_x, resolution_y))
done = False
totalError = 0
sumError = 0
numEpochs = 1

closest = None
distance = float("inf")
food_dim = None
features = []

def angle_between(p1, p2):
    # ang1 = np.arctan2(*p1[::-1])
    # ang2 = np.arctan2(*p2[::-1])
    # ang = np.rad2deg(ang1 - ang2)
    ang = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    return ang

def move(rect_dim, foods):
    # find which food is the shortest distance from the eater
    global closest, distance, food_dim, features

    if closest == None:
        distance = float("inf")
        food_dim = None
        features = []
    
        # calculate the distance between the eater and the food
        for food in foods:
            foods[food]["r"], foods[food]["g"] = 255, 128

            dist_to_food = math.hypot(foods[food]["rect_x"] - rect_dim["rect_x"], foods[food]["rect_y"] - rect_dim["rect_y"])

            angle_between_points = angle_between([rect_dim["rect_x"], rect_dim["rect_y"]], [foods[food]["rect_x"], foods[food]["rect_y"]])

            features.append(dist_to_food)
            features.append(angle_between_points)

            # print(dist_to_food)

            if dist_to_food < distance:
                distance = dist_to_food
                food_dim = foods[food]

        # print("-----------------------------------------")

    closest = food_dim
    food_dim["r"], food_dim["g"] = 128, 255

    # generate the position modifiers
    modifier_x, modifier_y = 0, 0
    modifier_network = model.predict(np.array([features]))
    modifier_network = modifier_network[0]

    max_index, _ = max(enumerate(modifier_network), key=lambda p: p[1])

    if max_index == 0:
        modifier_y = -1
    elif max_index == 1:
        modifier_x = -1
    elif max_index == 2:
        modifier_y = 1
    elif max_index == 3:
        modifier_x = 1

    # move the eater along the x axis
    rect_dim["rect_x"] += modifier_x

    # move the eater along the y axis
    rect_dim["rect_y"] += modifier_y

    # Train then network. This calculates what we expect the network to do, and compares it
    # to what the network actually decided to do
    targets = [0, 0, 0, 0]
    if (rect_dim["rect_x"] + rect_dim["rect_w"]) - (food_dim["rect_x"] + food_dim["rect_w"]) > 0:
        targets[1] = 1
    elif (rect_dim["rect_x"] + rect_dim["rect_w"]) - (food_dim["rect_x"] + food_dim["rect_w"]) < 0:
        targets[3] = 1

    if (rect_dim["rect_y"] + rect_dim["rect_h"]) - (food_dim["rect_y"] + food_dim["rect_h"]) > 0:
        targets[0] = 1
    elif (rect_dim["rect_y"] + rect_dim["rect_h"]) - (food_dim["rect_y"] + food_dim["rect_h"]) < 0:
        targets[2] = 1

    inputs = features

    model.fit(np.array([inputs]), np.array([targets]), epochs=1, verbose=2)
    return None
    # return nn.train(inputs, targets)

def new_food(food_dim):
    food_x = random.randint(0, resolution_x - food_dim["rect_w"])
    food_y = random.randint(0, resolution_y - food_dim["rect_h"])

    food_dim["rect_x"] = food_x
    food_dim["rect_y"] = food_y

# initialize food location
for food in foods: 
    new_food(foods[food])

# game loop
while not done:
    # check events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # refresh display
    pygame.display.flip()
    # screen.fill((0, 0, 0))
    screen.fill((255, 255, 255))

    # logic

    # eater
    eater = pygame.Rect(rect_dim["rect_x"], rect_dim["rect_y"], rect_dim["rect_w"], rect_dim["rect_h"])
    pygame.draw.rect(screen, (0, 128, 255), eater)

    # food
    for food_name in foods:
        food = pygame.Rect(foods[food_name]["rect_x"], foods[food_name]["rect_y"], foods[food_name]["rect_w"], foods[food_name]["rect_h"])
        pygame.draw.rect(screen, (foods[food_name]["r"], foods[food_name]["g"], foods[food_name]["b"]), food)

        if eater.colliderect(food):
            new_food(foods[food_name])
            closest = None

    error = move(rect_dim, foods)

    # show percent error
    # numEpochs += 1
    # sumError += abs(round(error[0][0] * 100, 2))
    # totalError = round(sumError / numEpochs, 2)
    # basicfont = pygame.font.SysFont(None, 20)
    # text = basicfont.render("Current error : {}%".format(abs(round(error[0][0] * 100, 2))), True, (255, 0, 0), (255, 255, 255))
    # text2 = basicfont.render("Average error over {} epochs: {}%".format(numEpochs, totalError), True, (255, 0, 0), (255, 255, 255))
    # screen.blit(text,(10,10))
    # screen.blit(text2,(10,30))

    # clock.tick(60)