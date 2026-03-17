""" The task is to categorize each face based on the emotion shown 
in the facial expression into one of seven categories 
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). 
"""

import os
import numpy as np
import cv2

class FER2013Dataset:
    def __init__(self):
        self.data_path = "fer2013" # relative path to the dataset folder
        self.data = []

    def test(self):
        self.load_data(split="train", expression="angry")
        print(f"Loaded {len(self.data)} samples for 'angry' expression in training set.")
        
    def load_data(self, split="train", expression="angry"):
        dir = self.data_path + f"/{split}/{expression}"
        if os.path.exists(dir):
            count = 0
            for filename in os.listdir(dir):
                if filename.endswith(".jpg"):
                    img = cv2.imread(os.path.join(dir, filename))
                    self.data.append((self.get_label(expression), img))
                count +=1
                if count % 1000 == 0:
                    print(f"Loaded {count} images from {dir}...")
        else:
            print(f"Directory {dir} does not exist.")


    def get_label(self, expression):
        expressions = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6}
        return expressions.get(expression, -1)

if __name__ == "__main__":
    dataset = FER2013Dataset()
    dataset.test()