from facenet_pytorch import InceptionResnetV1
from torchinfo import summary

# Load the InceptionResnetV1 model pre-trained on the VGGFace2 dataset
model = InceptionResnetV1(pretrained='vggface2').eval()

# Generate the summary
# Input shape: (Batch Size, Channels, Height, Width) -> (1, 3, 160, 160)
summary(model, input_size=(1, 3, 160, 160), col_names=("num_params", "mult_adds"))