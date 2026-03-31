import matplotlib.pyplot as plt


angry_bin = 958
disgust_bin = 111
fear_bin = 1024
happy_bin = 1774
sad_bin = 1247
surprise_bin = 831
neutral_bin = 1233

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
counts = [angry_bin, disgust_bin, fear_bin, happy_bin, sad_bin, surprise_bin, neutral_bin]
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray'])
plt.xlabel('Emotion')
plt.ylabel('Number of Images')
plt.title('Distribution of Images per Emotion in FER2013 Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('emotion_distribution.png')
plt.show()