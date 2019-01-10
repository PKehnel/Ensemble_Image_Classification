import numpy as np
import pandas as pd
from keras.applications import resnet50, vgg16, mobilenet, densenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import os
import matplotlib.pyplot as plt


# collect all pictures from the images folder. Resize them and add them to data

def collect_data():
    dir_path = 'images' #relativ link to the image folder
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        trueLabels.append(image_name.partition('.')[0])  # split to remove file extensions like ".jpg"
        data.append(img_array)
        plt.imshow(img)
        plt.show()


# for each model predict label and likelihood of each picture in Data

def predict_label():
    for i in range(len(modelEnsemble)):
        print(modelEnsemleArt[i], modelEnsemble[i])
        for x in range(len(data)):
            processed_image = modelEnsemleArt[i].preprocess_input(data[x])
            prediction = modelEnsemble[i].predict(processed_image)
            decode_prediction = decode_predictions(prediction, top=3)  # decode the results into a list of tuples (class, description, probability)
            print(decode_prediction)                                   # change top to x to get x highest predictions
            df.at[i, prediction_likelihood[2 * x]] = decode_prediction[0][0][1]     # save description
            df.at[i, prediction_likelihood[2 * x + 1]] = decode_prediction[0][0][2]     # save probability
            print(trueLabels[x])

# initialize models

resnet_model = resnet50.ResNet50(weights = 'imagenet')
vgg_model = vgg16.VGG16(weights = 'imagenet')
densenet_model = densenet.DenseNet169(weights = 'imagenet')
mobile_model = mobilenet.MobileNet(weights = 'imagenet')

modelEnsemble = [resnet_model, vgg_model, densenet_model]
# list of used models: resnet_model, vgg_model, mobile_model,  densenet_model
modelEnsemleArt = [resnet50, vgg16, densenet]
# list of model nets: resnet50, vgg16, mobilenet,   densenet
model_names = ['resnet50', 'vgg16', 'densenet']
# list of model Names: 'resnet50', 'vgg16', 'mobilenet','densenet'


data = []  # images to classify
trueLabels= [] # labels of images
extend = 'likelihood' # name for dataframe columns to store likelihood
prediction_likelihood = []  # store prediction and likelihood [house, 0.10% , room 0.2% ...]

collect_data()

# create a DataFrame in Form of:
# true Label:  house  | house_likelihood ; room  | room_likelihood
#    model 1:  apples | 0.2%             ; room  | 0.8%
#    model 2:  house  | 0.5%             ; room  | 0.7%

for x in trueLabels:
    prediction_likelihood.append(x)
    prediction_likelihood.append(x + extend)
predictions = np.zeros(shape=[len(modelEnsemble), len(prediction_likelihood)], dtype=str)
df = pd.DataFrame(predictions, columns=prediction_likelihood)

predict_label()

# replace index  with model names and save as pickle file with all model names
df['model_name'] = model_names
df.set_index('model_name', inplace=True)
df.to_pickle('predictionMix_for: ' + ''.join([x + ', ' for x in model_names])[:-2])

with pd.option_context('expand_frame_repr', False):
    print(df)
    print(trueLabels)
