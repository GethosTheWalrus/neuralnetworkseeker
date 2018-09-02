import pygame
import random
import math
import nn as neuralnetwork
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
        "rect_h": 10
    },
    "food_dim2": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10
    },
    "food_dim3": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10
    },
    "food_dim4": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10
    },
    "food_dim5": {
        "rect_x": 0,
        "rect_y": 0,
        "rect_w": 10,
        "rect_h": 10
    }

}

pygame.init()
screen = pygame.display.set_mode((resolution_x, resolution_y))
done = False
totalError = 0
sumError = 0
numEpochs = 1

# create the network object
nn = neuralnetwork.neuralNetwork()
nn.loadconfig("seeker")

def angle_between(p1, p2):
    # ang1 = np.arctan2(*p1[::-1])
    # ang2 = np.arctan2(*p2[::-1])
    # ang = np.rad2deg(ang1 - ang2)
    ang = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    return ang

def move(rect_dim, foods, nn):
    # find which food is the shortest distance from the eater
    distance = float("inf")
    food_dim = None
    features = []
    # Gets all the angles and distances between the eater and the foods
    for food_name in foods:
        # calculate the distance between the eater and the food
        dist_to_food = math.hypot(foods[food_name]["rect_x"] - rect_dim["rect_x"], foods[food_name]["rect_y"] - rect_dim["rect_y"])
        if dist_to_food < distance:
            distance = dist_to_food
            food_dim = foods[food_name]

        # get the angle between the eater and the food
        angle_between_points = angle_between([rect_dim["rect_x"], rect_dim["rect_y"]], [foods[food_name]["rect_x"], foods[food_name]["rect_y"]])

        # calculate the distance between the eater and the food before it moves
        dist_before = math.hypot(food_dim["rect_x"] - rect_dim["rect_x"], foods[food_name]["rect_y"] - foods[food_name]["rect_y"])

        # normalize features
        angle_between_points = abs(angle_between_points / 180)
        dist_before = 1 / (1 + np.exp(-dist_before)) 

        # add the normalized features to the feature vector
        features.append(angle_between_points)
        features.append(dist_before)

    # generate the position modifiers
    # modifier_network = nn.query([angle_between_points, dist_before])
    modifier_network = nn.query(features)
    if modifier_network[1] > modifier_network[3]:
        modifier_x = -1
    elif modifier_network[1] < modifier_network[3]:
        modifier_x = 1
    else:
        modifier_x = 0

    if modifier_network[0] > modifier_network[2]:
        modifier_y = -1
    elif modifier_network[0] < modifier_network[2]:
        modifier_y = 1
    else:
        modifier_y = 0

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

    return nn.train(inputs, targets)

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
            nn.saveconfig("seeker")
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
        pygame.draw.rect(screen, (255, 128, 128), food)

        if eater.colliderect(food):
            new_food(foods[food_name])

    error = move(rect_dim, foods, nn)

    # show percent error
    numEpochs += 1
    sumError += abs(round(error[0][0] * 100, 2))
    totalError = round(sumError / numEpochs, 2)
    basicfont = pygame.font.SysFont(None, 20)
    text = basicfont.render("Current error : {}%".format(abs(round(error[0][0] * 100, 2))), True, (255, 0, 0), (255, 255, 255))
    text2 = basicfont.render("Average error over {} epochs: {}%".format(numEpochs, totalError), True, (255, 0, 0), (255, 255, 255))
    screen.blit(text,(10,10))
    screen.blit(text2,(10,30))

    # clock.tick(60)