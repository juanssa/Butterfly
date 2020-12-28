# import the necessary packages
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import io
import os
import random
from imutils import paths
import imutils
import cv2
import time
import GLOBAL_VAR
norm_size = 32

#order_list = ["20 km","30 km","50 km","60 km","70 km","80 km", "100 km","120 km","no rebasar","interseccion Peligrosa", "alto","no hay paso","curva peligrosa a la izquierda","curva peligrosa a la derecha","doble curva","camino resbaloso","estrechamiento","peatones","niños cruzando","gire a der","gire a izq","recto","recto o der","recto o izq","siga por derecha","siga por izquierda","glorieta", "do_nothing", "None"]
order_list = ["Monarch", "Zebra Longwing", "Crimson-patched Longwing", "Common Buckeye", "American Copper",
              "Mournig Cloak", "Giant Swallowtail", "Cabbage White", "Red Admiral", "Painted Lady"]

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="model\\demo",help="path to trained model model")
    ap.add_argument("-i", "--image", type=str, default="images\\test",  help="path to input image")
    ap.add_argument("-s", "--show", action="store_true",  help="show predict image", default=True)
    ap.add_argument("-e", "--examples", type=str, default="examples", help="path to output examples directory")
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    
    print("[INFO] predicting...")
    imagePaths = list(paths.list_images(args["image"]))
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:15]

    for (i, imagePath) in enumerate(imagePaths):
        #load the image
        #image = cv2.imread(args["image"])
        image = cv2.imread(imagePath)  
        orig = image.copy()
         
        # pre-process the image for classification
        image = cv2.resize(image, (norm_size, norm_size))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
         
        # classify the input image
        result = model.predict(image)[0]
        #print (result.shape)
        proba = np.max(result)
        
        label = str(np.where(result == proba)[0])
        
        print(label[1:2])
        label_int = int(label[1:2])
        print(type(label_int))

        label_long = order_list[label_int]


        #label = "{}: {:.2f}%".format(label, proba * 100)
        label = "{}: {:.2f}%".format(label_long, proba * 100)
        print(label)
    
        #if args['show']:   
        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 255, 0), 2)   

        p = os.path.sep.join([args["examples"], "{}.png".format(i)])
        cv2.imwrite(p , output)

        while  True:
            cv2.imshow("Output", output)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            
        cv2.destroyAllWindows()

#python predict.py --model traffic_sign.model -i ../2.png -s
#python predict.py -m ..\model\traffic_sign_v2.model\ -i ..\images\test\5\0000.png -s
if __name__ == '__main__':
    args = args_parse()
    time_start = time.time()
    predict(args)
    time_end = time.time()
    print(time_end-time_start)