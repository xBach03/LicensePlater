import os
import cv2
from ultralytics import YOLO

# Define the image path
image_path = r'D:\VsCode\LicensePlate\dataset\test\images\xemay597_jpg.rf.f178bb96a2d50eb50ccb8eab78909800.jpg'

# Output directory for annotated image
output_directory = r'D:\VsCode\LicensePlate\detection_results'  

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Load the YOLO model
model_path = r'D:\VsCode\LicensePlate\runs\detect\train9\weights\last.pt'
model = YOLO(model_path)

threshold = 0.1

# Read the image
image = cv2.imread(image_path)

# Perform inference
results = model(image)[0]

# Iterate over detected objects
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Write the annotated image to the output directory
output_path = os.path.join(output_directory, 'annotated_image.jpg')
cv2.imwrite(output_path, image)
print(f'Annotated image saved: {output_path}')
cv2.imshow("detected", image)
cv2.waitKey(0)

