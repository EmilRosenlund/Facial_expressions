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
    import random
    def augmentations(img):
        # Squeeze (random horizontal/vertical scaling)
        squeeze_factor = random.uniform(0.7, 1.0)
        squeeze_img = img.resize((int(img.width * squeeze_factor), img.height))
        squeeze_img = squeeze_img.resize((img.width, img.height))
        # Rotate
        rotate_img = img.rotate(random.uniform(-20, 20))
        # Add Gaussian noise
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, np_img.shape)
        noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        return [squeeze_img, rotate_img, noisy_img]

    for split in ["train", "test"]:
        for expression in ["angry", "disgust", "sad", "surprise", "neutral", "fear", "happy"]:
            data = dataset.load_data(split=split, expression=expression)
            batch_size = 32
            all_embeddings = []
            batch = []
            for idx, (label, img) in enumerate(data):
                imgs_to_embed = [img.convert('RGB')]
                imgs_to_embed.extend(augmentations(img.convert('RGB')))
                for aug_img in imgs_to_embed:
                    preprocessed_img = preprocess(aug_img)
                    batch.append(preprocessed_img)
                    if len(batch) == batch_size:
                        batch_tensor = torch.stack(batch)
                        with torch.no_grad():
                            embeddings = embedding_creator.get_embedding(batch_tensor)
                        all_embeddings.append(embeddings.cpu().numpy())
                        batch = []
                if idx % 100 == 0:
                    print(f"Processed {idx} images for {split} set, {expression} expression...")
            if batch:
                batch_tensor = torch.stack(batch)
                with torch.no_grad():
                    embeddings = embedding_creator.get_embedding(batch_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            os.makedirs("embeddings_ResNet50", exist_ok=True)
            save_path = f"embeddings_ResNet50/embeddings_{split}_{expression}.npy"
            np.save(save_path, all_embeddings)
            print(f"Saved embeddings for {split} set, {expression} expression to {save_path}.")