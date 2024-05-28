from typing import Union
from fastapi import FastAPI, File, UploadFile, Form, WebSocket
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from eigenfaces_model import read_images, as_row_matrix, pca, project, predict, subplot
from new_eigenfaces import show_dataset, detect_face
import ffmpeg
import uvicorn
import base64
import imghdr
import cv2
from pydantic import BaseModel
from fastapi import WebSocket
import json

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class FaceRecognitionRequest(BaseModel):
    file: str
    npk: str

def get_image_extension_from_base64(base64_string):
    try:
        
        # Decode the base64 string
        decoded_data = base64.b64decode(base64_string)
        
        # Use imghdr to determine the image type from the header
        image_type = imghdr.what(None, decoded_data)
        
        if image_type:
            # Return the appropriate extension based on the image type
            return f".{image_type}"
        else:
            # If imghdr couldn't determine the image type, return None
            return None
    except Exception as e:
        print(f"Error getting image extension: {e}")
        return None
    
def decode_base64_to_image(base64_string, output_path):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img

# cv2 videocapture
def vc_extract_images_from_video(video_path, output_directory, base_name, frame_interval=30, max_frames=72):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the output directory if it doesn't exist

    cap = cv2.VideoCapture(video_path)  # Open the video file
    i = 0
    frame_count = 0

    while cap.isOpened() and i < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_directory, f"{base_name}_{i:04d}.jpg"), frame)
            i += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

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
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        # Draw text (name) on the image
        # Draw red text (name) on the image with bigger font size
        cv2.putText(img, "Alferdian", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

        
        # Crop the face region from the original image
        # cropped_face = img[y:y+h, x:x+w]
        
        # Save the cropped face as a separate image
        output_face_path = opt + f"{npk}.jpeg"
        # cv2.imwrite(output_face_path, cropped_face)
        
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

# new model eigenfaces
@app.post("/predict-photo")
async def predict_photo(file: str = Form(...), npk : str = Form(...)):
    try:
        extension = get_image_extension_from_base64(file)

        upload_folder_path = os.path.join(os.getcwd(), UPLOAD_FOLDER)
        file_path = os.path.join(upload_folder_path, npk + extension)

        save_uploaded_file(file, file_path) 


        dataset_folder = 'training-images/'

        names = []
        images = []

        for folder in os.listdir(dataset_folder):
            for name in os.listdir(os.path.join(dataset_folder, folder))[:8]: # limit only 70 face per class
                if name.find(".png") > -1 :
                    img = cv2.imread(os.path.join(dataset_folder + folder, name))
                    images.append(img)
                    names.append(folder)

        # for folder in os.listdir(dataset_folder):
        #     folder_path = os.path.join(dataset_folder, folder)
        #     if not os.path.isdir(folder_path):
        #         continue  # Skip if it's not a directory

        #     for name in os.listdir(folder_path):  # Limit only 70 faces per class if needed
        #         if name.endswith(".jpg"):  # Ensure only .jpg files are processed
        #             img_path = os.path.join(folder_path, name)
        #             img = cv2.imread(img_path)
        #             if img is not None:  # Ensure the image was read successfully
        #                 images.append(img)
        #                 names.append(folder)
        
        labels = np.unique(names)
        for label in labels:
            ids = np.where(label== np.array(names))[0]
            images_class = images[ids[0] : ids[-1] + 1]
            # show_dataset(images_class, label)



        croped_images = []
        for i, img in enumerate(images) :
            img = detect_face(img, i)
            if img is not None :
                croped_images.append(img)
            else :
                del names[i]
        
        for label in labels:
            ids = np.where(label== np.array(names))[0]
            images_class = croped_images[ids[0] : ids[-1] + 1] # select croped images for each class
            # show_dataset(images_class, label)

        name_vec = np.array([np.where(name == labels)[0][0] for name in names])

        # model = cv2.face.EigenFaceRecognizer_create()
        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(croped_images, name_vec)

        model.save("eigenface.yml")

        model.read("eigenface.yml")

        #cv2.model


        # labels = get_folder_names(dataset_folder)

        path = "uploads/" + npk + extension

        img = cv2.imread(path)
        img = detect_face(img, 0)

        idx, confidence = model.predict(img)

        print("FOUND IDX" , idx)

        print("Found: ", labels[idx])
        print("Confidence: ", confidence)

        # check if labels[idx] is null and confidence is null then return status code 400
        if labels[idx] is None and confidence is None:
            return {"message": "No faces detected", "contains_face": False}

        return {"filename": npk, "file_path": file_path, "predict": labels[idx], "confidence": confidence}
        # return {"filename": npk, "file_path": file_path}
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return False
    

def save_uploaded_file(file, destination):
    try:
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
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def get_folder_names(directory):
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return folder_names

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Parse JSON data
            json_data = json.loads(data)
            base64_image = json_data.get("image")
            npk = json_data.get("npk")
            print(base64_image)

            extension = get_image_extension_from_base64(base64_image)
            print(extension)


            upload_folder_path = os.path.join(os.getcwd(), UPLOAD_FOLDER)
            file_path = os.path.join(upload_folder_path, npk + extension)

            save_uploaded_file(base64_image, file_path) 

            dataset_folder = 'training-images/'

            model = cv2.face.EigenFaceRecognizer_create()

            model.read("eigenface.yml")


            labels = get_folder_names(dataset_folder)

            path = "uploads/" + npk + extension

            img = cv2.imread(path)
            img = detect_face(img, 0)

            if img is None:
                await websocket.send_json({"message": "No faces detected", "contains_face": False})
                break

            idx, confidence = model.predict(img)
            

            print("Found: ", labels[idx])
            print("Confidence: ", confidence)

            # check if labels[idx] is null and confidence is null then return status code 400
            if labels[idx] is None and confidence is None:
                response = {"message": "", "contains_face": False}
                await websocket.send_json(response)

            
            # if npk != label[idx] return response 400
            if npk != labels[idx]:
                response = { "message": "NPK not found in the database", "npk":npk, "predict": labels[idx], "confidence": confidence}
                await websocket.send_json(response)
                break

            response = {"filename": npk, "file_path": file_path, "predict": labels[idx], "confidence": confidence}
            await websocket.send_json(response)

    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})


async def process_data(base64_image: str, npk_string: str):
    try:
        # Decode base64 image to bytes
        image_data = base64.b64decode(base64_image)
        # Convert bytes to numpy array
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        # Convert numpy array to PIL Image
        image = Image.open(io.BytesIO(image_np))
        
        # Save the image to disk
        image.save("received_image.jpg")
        
        # Process NPK string (you can implement your logic here)
        print("Received NPK string:", npk_string)
        
        return {"status": "success", "message": "Image and NPK string processed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


    