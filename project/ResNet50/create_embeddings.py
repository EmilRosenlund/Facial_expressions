#from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms

import numpy as np
import os

import torchvision.models as models

class create_resnet_embeddings:
    def __init__(self):
        # Load standard ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        # Fjern det sidste lag (fc) for at få 2048-d embeddings
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

    def get_embedding(self, batch_tensor):
        with torch.no_grad():
            # Output er [batch, 2048, 1, 1], så vi squeezer de sidste dimensioner
            embedding = self.model(batch_tensor)
            return embedding.view(embedding.size(0), -1) 
        
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

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataloader import FER2013Dataset
    dataset = FER2013Dataset()
    embedding_creator = create_resnet_embeddings()
    for split in ["train", "test"]:
        for expression in ["angry"]:
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
                        embeddings = embedding_creator.get_embedding(batch_tensor)
                    all_embeddings.append(embeddings.cpu().numpy())
                    batch = []
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            os.makedirs("embeddings_ResNet50", exist_ok=True)
            save_path = f"embeddings_ResNet50/embeddings_{split}_{expression}.npy"
            
            np.save(save_path, all_embeddings)
            print(f"Saved embeddings for {split} set, {expression} expression to {save_path}.")