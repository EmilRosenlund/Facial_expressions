from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

import numpy as np
import os


class create_embeddings:
    def __init__(self):
        self.image_size = 160
        self.margin = 0

        # Create face detection pipeline
        self.mtcnn = MTCNN(image_size=self.image_size, margin=self.margin)

        # Create an inception resnet (in eval mode)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()


    def get_embedding(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = preprocess(image.convert('RGB'))
        img_embedding = self.resnet(image.unsqueeze(0))
        return img_embedding
        
    def save_embedding(self, embedding, save_path):
        if embedding is not None:
            numpy_embedding = embedding.detach().cpu().numpy()
            np.save(save_path, numpy_embedding)

if __name__ == "__main__":    
    # load all images from the dataset and create embeddings
    from dataloader import FER2013Dataset
    dataset = FER2013Dataset(debug=False)
    for split in ["train", "test"]:
        for expression in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            embedding_array = []
            data = dataset.load_data(split=split, expression=expression)
            for idx, (label, img) in enumerate(data):
                embedding_creator = create_embeddings()
                embedding = embedding_creator.get_embedding(img)
                if embedding is not None:
                    embedding_array.append(embedding.detach().cpu().numpy())
            embedding_array = np.array(embedding_array)
            save_path = f"embeddings_{split}_{expression}.npy"
            os.makedirs("embeddings", exist_ok=True)
            np.save(save_path, embedding_array)
            print(f"Saved embeddings for {split} set, {expression} expression to {save_path}.")
                

  