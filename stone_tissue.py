import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to("cpu")  # Use "cuda" if GPU available

predictor = SamPredictor(sam)


image = cv2.imread("stone_tissue.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)


input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
input_label = np.array([1])  # 1 means "foreground"


masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)


plt.imshow(image_rgb)
plt.imshow(masks[0], alpha=0.5)
plt.axis('off')
plt.title("Segmented Stone Tissue")
plt.show()


combined = image_rgb.copy()
combined[masks[0]] = [255, 0, 0]  # Apply red mask
cv2.imwrite("annotated_stone_tissue.jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
print("Saved as annotated_stone_tissue.jpg")
