import os
import sys
import time

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.transforms import functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import VGGFace2WithMLP


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HIDDEN_SIZE = 128
NUM_CLASSES = 7
TARGET_FPS = 20
MODEL_PATH = "best_model_v5.pth"


def load_model(device: torch.device) -> torch.nn.Module:
	model = VGGFace2WithMLP(HIDDEN_SIZE, NUM_CLASSES)
	checkpoint = torch.load(MODEL_PATH, map_location=device)

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint

	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	return model


def preprocess_face_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
	face_tensor = face_tensor.float()
	if face_tensor.max() > 1.0:
		face_tensor = face_tensor / 255.0
	face_tensor = TF.rgb_to_grayscale(face_tensor, num_output_channels=3)
	face_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(face_tensor)
	return face_tensor


def main() -> None:
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	print(f"Loading model from: {MODEL_PATH}")

	model = load_model(device)

	mtcnn = MTCNN(
		image_size=160,
		margin=20,
		min_face_size=40,
		keep_all=True,
		post_process=False,
		device=device,
	)

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

	if not cap.isOpened():
		raise RuntimeError("Could not open webcam.")

	target_frame_time = 1.0 / TARGET_FPS
	print("Webcam started. Press 'q' to quit.")

	try:
		while True:
			loop_start = time.time()

			ok, frame_bgr = cap.read()
			if not ok:
				print("Failed to read frame from webcam.")
				break

			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			boxes, _ = mtcnn.detect(frame_rgb)
			cropped_faces, probs = mtcnn(frame_rgb, return_prob=True)

			if boxes is not None and cropped_faces is not None:
				face_tensors = []
				valid_boxes = []

				for i, box in enumerate(boxes):
					x1, y1, x2, y2 = box.astype(int)
					x1 = max(0, x1)
					y1 = max(0, y1)
					x2 = min(frame_rgb.shape[1], x2)
					y2 = min(frame_rgb.shape[0], y2)

					if x2 <= x1 or y2 <= y1:
						continue
					if i >= len(cropped_faces):
						continue

					face_tensors.append(preprocess_face_tensor(cropped_faces[i]))
					valid_boxes.append((x1, y1, x2, y2, probs[i] if probs is not None else None))

				if face_tensors:
					batch = torch.stack(face_tensors).to(device)
					with torch.no_grad():
						logits = model(batch)
						pred_idx = torch.argmax(logits, dim=1)
						pred_conf = torch.softmax(logits, dim=1).max(dim=1).values

					for i, (x1, y1, x2, y2, det_prob) in enumerate(valid_boxes):
						cls = CLASS_NAMES[int(pred_idx[i].item())]
						conf = float(pred_conf[i].item()) * 100.0
						det_txt = "" if det_prob is None else f" det:{float(det_prob) * 100.0:.1f}%"
						label = f"{cls} {conf:.1f}%{det_txt}"

						cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
						cv2.putText(
							frame_bgr,
							label,
							(x1, max(20, y1 - 8)),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.6,
							(0, 220, 0),
							2,
							cv2.LINE_AA,
						)

			elapsed = time.time() - loop_start
			fps = 1.0 / elapsed if elapsed > 0 else 0.0
			cv2.putText(
				frame_bgr,
				f"FPS: {fps:.1f} (target {TARGET_FPS})",
				(20, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(60, 200, 255),
				2,
				cv2.LINE_AA,
			)

			cv2.imshow("Facial Expression Live Inference", frame_bgr)

			# Throttle loop to target FPS.
			sleep_time = target_frame_time - (time.time() - loop_start)
			if sleep_time > 0:
				time.sleep(sleep_time)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
