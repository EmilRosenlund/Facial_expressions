from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms

import numpy as np
import os


class create_embeddings:
    def __init__(self):
        self.image_size = 160
        self.margin = 0

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
    """ from dataloader import FER2013Dataset
    dataset = FER2013Dataset(debug=False)
    embedding_creator = create_embeddings()
    for split in ["train", "test"]:
        for expression in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            embedding_array = []
            data = dataset.load_data(split=split, expression=expression)
            for idx, (label, img) in enumerate(data):
                embedding = embedding_creator.get_embedding(img)
                if embedding is not None:
                    embedding_array.append(embedding.detach().cpu().numpy()) # Convert tensor to numpy array cpu means move to cpu and detach means remove from computation graph
                if idx % 100 == 0: print(f"Processed {idx} images for {split} set, {expression} expression...")
            embedding_array = np.array(embedding_array)
            save_path = f"embeddings_{split}_{expression}.npy"
            os.makedirs("embeddings", exist_ok=True)
            np.save(save_path, embedding_array)
            print(f"Saved embeddings for {split} set, {expression} expression to {save_path}.")
                 """

    preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])            

    from dataloader import FER2013Dataset
    dataset = FER2013Dataset()
    embedding_creator = create_embeddings()
    for split in ["train", "test"]:
        for expression in [ "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            data = dataset.load_data(split=split, expression=expression)
            batch_size = 32
            all_embeddings = []
            batch = []
            for idx, (label, img) in enumerate(data):
                preprocessed_img = preprocess(img.convert('RGB'))
                batch.append(preprocessed_img)
                if len(batch) == batch_size or idx == len(data) - 1:
                    batch_tensor = torch.stack(batch)
                    with torch.no_grad():
                        embeddings = embedding_creator.resnet(batch_tensor)
                    all_embeddings.append(embeddings.cpu().numpy())
                    batch = []
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            save_path = f"embeddings_{split}_{expression}.npy"
            os.makedirs("embeddings", exist_ok=True)
            np.save(save_path, all_embeddings)
            print(f"Saved embeddings for {split} set, {expression} expression to {save_path}.")