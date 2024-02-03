from typing import Union


from fastapi import FastAPI, File, UploadFile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from eigenfaces_model import read_images, as_row_matrix, pca, project, predict, subplot

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def read_root():
    print("HELo")
    return {"Hello": "World"}

@app.post("/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    upload_folder_path = os.path.join(os.getcwd(), UPLOAD_FOLDER)
    file_path = os.path.join(upload_folder_path, file.filename)
    save_uploaded_file(file, file_path)

    [X, y] = read_images()

    [eigenvalues, eigenvectors, mean] = pca (as_row_matrix(X), y)
    
    projections = []
    for xi in X:
        projections.append(project (eigenvectors, xi.reshape(1 , -1) , mean))

    image = Image.open(file_path)
    image = image.convert ("L")

    DEFAULT_SIZE = [250, 250]
    
    if (DEFAULT_SIZE is not None ):
        image = image.resize (DEFAULT_SIZE , Image.ANTIALIAS )
    test_image = np. asarray (image , dtype =np. uint8 )
    predicted = predict(eigenvectors, mean , projections, y, test_image)




    return {"filename": file.filename, "content_type": file.content_type, "file_path": file_path, "predict": y[predicted]}

def save_uploaded_file(file, destination):
    with open(destination, "wb") as f:
        f.write(file.file.read())

