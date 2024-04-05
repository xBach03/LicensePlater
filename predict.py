import cv2 as cv
from visualize import readLicensePlate
from ultralytics import YOLO

# # Define the image path
# image_path = r'D:\VsCode\dataset(origin)\dataset\test\images\xemay1223_jpg.rf.c27d717c455451a3b01e7283324a1faf.jpg'

# Load the YOLO model
model_path = r'D:\VS Code\LicensePlater\runs\detect\train2\weights\last.pt'
model = YOLO(model_path)

# Read image 
# image = cv.imread(image_path)

# result = model(source=0, show=True, conf=0.3, save=False)


# Initialize camera
cap = cv.VideoCapture(0)  # 0 represents the default camera (you can change it if needed)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform detection
    result = model(frame)[0]

    # Assuming you have access to the detected region of the license plate
    for detection in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        # Crop license plate
        licensePlateCrop = frame[int(y1):int(y2), int(x1):int(x2), :]
        
        # Process license plate
        licensePlateCropGray = cv.cvtColor(licensePlateCrop, cv.COLOR_BGR2GRAY)
        # _, licensePlateCropThresh = cv.threshold(licensePlateCropGray, 64, 255, cv.THRESH_BINARY_INV)


        # Read license plate
        licensePlateText, confidenceScore = readLicensePlate(licensePlateCropGray)
    

        # Draw rectangle detector
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Print license plate
        cv.putText(frame, licensePlateText, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        # Display the resulting frame
    cv.imshow('Frame', frame)

    # Press 'q' to exit
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv.destroyAllWindows()

# Read license plate
# Perform inference 
# results = model(image)[0]
# for detection in results.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = detection

#     # Crop license plate
#     licensePlateCrop = image[int(y1):int(y2), int(x1):int(x2), :]
    
#     # Process license plate
#     licensePlateCropGray = cv.cvtColor(licensePlateCrop, cv.COLOR_BGR2GRAY)
#     _, licensePlateCropThresh = cv.threshold(licensePlateCropGray, 110, 255, cv.THRESH_BINARY_INV)


#     # Read license plate
#     licensePlateText, confidenceScore = readLicensePlate(licensePlateCropThresh)

#     # Draw rectangle detector
#     cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#     # Print license plate
#     cv.putText(image, licensePlateText, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

# # Display the annotated image
# cv.imshow("License Plate Detection", image)
# cv.waitKey(0)
# cv.destroyAllWindows() 



    