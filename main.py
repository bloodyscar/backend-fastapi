from typing import Union
from fastapi import FastAPI, File, UploadFile, Form
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from eigenfaces_model import read_images, as_row_matrix, pca, project, predict, subplot
import ffmpeg
import uvicorn
import base64
import imghdr
import cv2

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_image_extension_from_base64(base64_string):
    # Decode the base64 string
    decoded_data = base64.b64decode(base64_string, validate=True)
    
    # Use imghdr to determine the image type from the header
    image_type = imghdr.what(None, decoded_data)
    
    if image_type:
        # Return the appropriate extension based on the image type
        return f".{image_type}"
    else:
        # If imghdr couldn't determine the image type, return None
        return None
    
def decode_base64_to_image(base64_string, output_path):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))

def extract_images_from_video(video_path, output_directory, npk, options={}):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the output directory if it doesn't exist
    ffmpeg.input(video_path)\
        .output(os.path.join(output_directory, npk+'_%03d.png'), vf='select=not(mod(n\\,30))', vsync='vfr')\
        .run()

@app.get("/")
def read_root():
    print("HELo")
    return {"Hello": "World"}

@app.post("/register_face")
async def save_image(npk : str = Form(...), video: UploadFile = File(...), ):
    try:
        with open(f"uploads/{video.filename}", "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Usage
        project_root = os.getcwd(); 
        video_path = os.path.join(project_root, "uploads", video.filename)
        output_directory = "training-images/" + npk

        extract_images_from_video(video_path, output_directory, npk)
        print('Images extracted successfully.')

    # Delete the uploaded file after successful image extraction
        os.remove(video_path)
        return {"message": "Image saved successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/check_face")
async def check_face(npk : str = Form(...)):
    try:
        if os.path.exists("training-images/" + npk):
            print("Folder exists")
            return {"message": "Data face sudah ada"}
        else:
            print("Folder does not exist")
            return {"message": "Data face belum ada"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload-photo")
async def upload_photo(file: str = Form(...), npk : str = Form(...)):
    
    extension = get_image_extension_from_base64(file)

    upload_folder_path = os.path.join(os.getcwd(), UPLOAD_FOLDER)
    file_path = os.path.join(upload_folder_path, npk + extension)

    save_uploaded_file(file, file_path) 

    ## start detect face
    img = cv2.imread(file_path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    opt = "uploads/"

    for i, (x, y, w, h) in enumerate(face):
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # Crop the face region from the original image
        cropped_face = img[y:y+h, x:x+w]
        
        # Save the cropped face as a separate image
        output_face_path = opt + f"{npk}.jpeg"
        cv2.imwrite(output_face_path, cropped_face)
        
        print(f"Face {i+1} saved successfully to:", output_face_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,10))
    plt.imshow(img_rgb)
    plt.axis('off') 

    # Save the image
    output_path = os.path.join("uploads/", "detected_faces.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Check if any faces are detected
    if len(face) > 0:
        [X, y] = read_images()

        [eigenvalues, eigenvectors, mean] = pca (as_row_matrix(X), y)
        
        projections = []
        for xi in X:
            projections.append(project (eigenvectors, xi.reshape(1 , -1) , mean))

        image = Image.open(file_path)
        image = image.convert ("L")

        DEFAULT_SIZE = [250, 250]
        
        if (DEFAULT_SIZE is not None ):
            image = image.resize (DEFAULT_SIZE , Image.LANCZOS )
        test_image = np. asarray (image , dtype =np. uint8 )
        predicted = predict(eigenvectors, mean , projections, y, test_image)

        return {"filename": npk, "file_path": file_path, "predict": y[predicted]}
    else:
        return {"message": "No faces detected", "output": output_path, "contains_face": False}
    
    ## end detect face

    

def save_uploaded_file(file, destination):
    # with open(destination, "wb") as image_file:
    #     image_file.write(base64.b64decode(file))    
    # Decode the base64-encoded image data
    image_data = base64.b64decode(file)
    
    # Convert the image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale image
    cv2.imwrite(destination, gray_img)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)