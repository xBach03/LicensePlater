import cv2 as cv
from visualize import readLicensePlate
from ultralytics import YOLO

# # Define the image path
# image_path = r'D:\VsCode\dataset(origin)\dataset\test\images\xemay1223_jpg.rf.c27d717c455451a3b01e7283324a1faf.jpg'
# # Define the image path
# image_path = r'D:\VsCode\dataset(origin)\dataset\test\images\xemay1223_jpg.rf.c27d717c455451a3b01e7283324a1faf.jpg'

# # Load the YOLO model
# model_path = r'D:\VsCode\LicensePlate\runs\detect\train2\weights\last.pt'
# model = YOLO(model_path)

# # Read image 
# image = cv.imread(image_path)

# # result = model(source=1, show=True, conf=0.3, save=False)

# # Perform inference 
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

# Process image and detect license plates
def process_image(image_path, model_path):
    # Load the YOLO model
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
        licensePlateText, confidenceScore = readLicensePlate(licensePlateCrop)

        # Draw rectangle around detected plate
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Print license plate
        if licensePlateText:
            cv.putText(image, licensePlateText, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return image, licensePlateText

    