import cv2
import numpy as np

from datetime import datetime
import random


def b_preprocessing(sequence, config):

    # handle sequences < len 4 as [1, 2, 3] -> [1, 1, 2, 3]
    while len(sequence) < 4:
        sequence.appendleft(sequence[1])

    output = []

    for i in range(1, config.agent_history_length + 1):
        output.append(preprocessing(sequence[-i-1], sequence[-1]))

    return np.array(output)


def preprocessing(previous_frame, current_frame):

    # select maximum value for each pixel colour over previous and current frame to remove flickering
    image = np.maximum(previous_frame, current_frame, dtype=np.float32)
    #image = np.concatenate((previous_frame[..., np.newaxis], current_frame[..., np.newaxis]), axis=3, dtype=np.float32).max(axis=3)

    # convert to gray-scale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # rescale image to 84 x 84
    image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)

    #cv2.imwrite(r'C:\Users\Jannik\Desktop\deep-q-network\img\run_img_' + str(random.randint(1, 10000)) + ".jpeg", image)

    return image
