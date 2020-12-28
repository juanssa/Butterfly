# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# importar los paquetes necesarios
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
from red.butterfly import ButterflyNet



def args_parse():
    # construir el argumento analizar y analizar los argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", type=str, default="images\\Test", help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", type=str, default="images\\Train\\", help="path to input dataset_train")
    ap.add_argument("-m", "--model", type=str, default="model\\demo", help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="model\\result_plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


# inicializar el número de épocas para entrenar, tasa de aprendizaje inicial,
# y tamaño de lote
EPOCHS = 200
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 10
norm_size = 32


def load_data(path):
    print("[INFO] loading images...", path)
    data = []
    labels = []
    # tomar las rutas de las imágenes y mezclarlas aleatoriamente
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # bucle sobre las imágenes de entrada
    cont = 0
    for imagePath in imagePaths:
    	cont = cont + 1
    	#print(cont, imagePath)

        # cargar la imagen, preprocesarla y almacenarla en la lista de datos
    	image = cv2.imread(imagePath)
    	image = cv2.resize(image, (norm_size, norm_size))
    	image = img_to_array(image)
    	data.append(image)

        # extraer la etiqueta de clase de la ruta de la imagen y actualizar la
        # lista de etiquetas
    	label = int(imagePath.split(os.path.sep)[-2])
    	
    	labels.append(label)
    
    # escalar las intensidades de píxeles sin procesar al rango [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convertir las etiquetas de números enteros a vectores
    labels = to_categorical(labels, num_classes=CLASS_NUM)                         
    return data, labels
    

def train(aug, trainX, trainY, testX, testY, args):
    # inicializar el modelo
    print("[INFO] compiling model...")
    model = ButterflyNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # entrenar a la red
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), 
                            steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, 
                            verbose=1)

    # guardar el modelo en el disco
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    # trazar la pérdida de entrenamiento y la precisión
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # en tensorflow 2.0 acc cambia por val_accuracy
    # y val_acc cambia por val_accuracy
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    

if __name__=='__main__':

    args = args_parse()
    
    train_file_path = args["dataset_train"]
    test_file_path = args["dataset_test"]
    trainX, trainY = load_data(train_file_path)
    testX, testY = load_data(test_file_path)
    # construir el generador de imágenes para el aumento de datos
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    train(aug, trainX, trainY, testX, testY, args)