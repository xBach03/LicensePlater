import cv2 as cv
from visualize import readLicensePlate
from ultralytics import YOLO

# Define the image path
image_path = r'D:\VsCode\dataset(origin)\dataset\test\images\xemay581_jpg.rf.5cc25fed49cebb0f59cb884420bedb9a.jpg'

# Load the YOLO model
model_path = r'D:\VsCode\dataset(origin)\runs\detect\train9\weights\last.pt'
model = YOLO(model_path)

# Read image 
image = cv.imread(image_path)

# Perform inference 
results = model(image)[0]
for detection in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection

    # Crop license plate
    licensePlateCrop = image[int(y1):int(y2), int(x1):int(x2), :]
    
    # Process license plate
    licensePlateCropGray = cv.cvtColor(licensePlateCrop, cv.COLOR_BGR2GRAY)
    _, licensePlateCropThresh = cv.threshold(licensePlateCropGray, 110, 255, cv.THRESH_BINARY_INV)

    # Read license plate
    licensePlateText, confidenceScore = readLicensePlate(licensePlateCropThresh)
    print(licensePlateText)

    # Draw rectangle detector
    cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Print license plate
    cv.putText(image, licensePlateText, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

# Display the annotated image
cv.imshow("License Plate Detection", image)
cv.waitKey(0)
cv.destroyAllWindows() 

    