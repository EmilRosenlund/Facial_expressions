import matplotlib.pyplot as plt
import numpy as np


class_counts = np.array([3995, 436, 4097, 7215, 4965, 4830, 3171])

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
counts = class_counts.tolist()
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray'])
plt.xlabel('Emotion')
plt.ylabel('Number of Images')
plt.title('Distribution of Images per Emotion in FER2013 Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('emotion_distribution.png')
plt.show()