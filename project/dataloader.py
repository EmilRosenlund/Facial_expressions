""" The task is to categorize each face based on the emotion shown 
in the facial expression into one of seven categories 
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). 
"""

import os
import numpy as np
import PIL.Image as Image

class FER2013Dataset:
    def __init__(self, debug=True):
        self.data_path = "/ceph/home/student.aau.dk/cu73wa/Facial_expressions/fer2013" # relative path to the dataset folder
        self.debug = debug

    def test(self):
        data = self.load_data(split="train", expression="angry")
        print(f"Loaded {len(data)} samples for 'angry' expression in training set.")

    def load_data(self, split="train", expression="angry"):
        dir = self.data_path + f"/{split}/{expression}"
        data = []
        if os.path.exists(dir):
            count = 0
            for filename in os.listdir(dir):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(dir, filename)
                    with Image.open(img_path) as img:
                        img_converted = img.copy()  # Copy image data to memory, then close file
                    data.append((self.get_label(expression), img_converted))
                count += 1
                if self.debug and count % 1000 == 0:
                    print(f"Loaded {count} images from {dir}...")
        else:
            print(f"Directory {dir} does not exist.")
        return data

    def get_label(self, expression):
        expressions = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6}
        return expressions.get(expression, -1)

    def load_embeddings(self, split="train", expression="angry"):
        embedding_path = f"embeddings/embeddings_{split}_{expression}.npy"
        if os.path.exists(embedding_path):
            return np.load(embedding_path)
        else:
            print(f"Embedding file {embedding_path} does not exist.")
            return None

if __name__ == "__main__":
    dataset = FER2013Dataset()
    dataset.test()