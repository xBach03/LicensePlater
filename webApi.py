import os
import cv2  # Ensure cv2 is imported
from random import random
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from predict import process_image

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "static"
# Load YOLOv8 model with the provided absolute path
weights_path = r"D:\VsCode\LicensePlate\runs\detect\train2\weights\best.pt"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at {weights_path}")

yolov8_model = YOLO(weights_path)

save_path = ""

@app.route("/", methods=['GET', 'POST'])
def home_page():
    global save_path
    if request.method == "POST":
        try:
            # Read posted file
            image = request.files.get('file')
            if image:
                # Save file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                save_path = path_to_save
                print("Save =", path_to_save)
                image.save(path_to_save)

                # Process image and detect license plates
                processed_image, licensePlateText = process_image(path_to_save, weights_path)
                
                # Save the image with bounding boxes
                result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{image.filename}")
                cv2.imwrite(result_image_path, processed_image)

                if licensePlateText:
                    return render_template("index.html", user_image=f"result_{image.filename}", rand=str(random()), msg="You are checked in!", ndet = 1)
                else:
                    return render_template('index.html', user_image=f"result_{image.filename}", rand=str(random()), msg="Cannot extract content from number plate", ndet = 1)
            else:
                # If no file exists, requesting user to upload file
                return render_template('index.html', msg='Choose file to upload', ndet = 0)

        except Exception as ex:
            # Print error
            print(ex)
            return render_template('index.html', msg='Cannot recognize number plate(s)', ndet = 0)

    else:
        # If GET -> render index page
        return render_template('index.html', ndet=0)

@app.route('/get-license', methods = ['GET'])
def get_license_plate():
    # global licensePlateContent
    # print("licensePlateContent: " + licensePlateContent)
    # if licensePlateContent:
    #     licenseResult = licensePlateContent
    #     licensePlateContent = ""  # Reset the content after reading
    #     return jsonify({"status": "complete", "licenseResult": licenseResult})
    # else:
    #     return jsonify({"status": "failed"})
    processed_image, license_plate_content = process_image(save_path, weights_path)
    return jsonify({"licensePlateContent": license_plate_content, "status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6868, debug=True)