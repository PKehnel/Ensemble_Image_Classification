# Ensemble_Image_Classification
creating an ensemble from multiple pretrained networks, to see if this achieves higher results

# networkPredictions: 

choose multiple pretrained models:
resnet_model, vgg_model, mobile_model,  densenet_model

populate the folder: 'images' with the pictures to classify. 

True labels are currently the names of images, or just input them as array (only needed for evaluation)

output: dataframe saved as pickle file 

| Label |  house  |  house_likelihood  |  room  |  room_likelihood  |
| ------------- | ------------- | ------------- | ------------- | ------------- |
|  model 1:  | apples  |  0.2%  |  room  |  0.8%  |
|  model 2:  | house  |  0.5%  |  room  |  0.7%  |

# ensemble
not complete

load the dataframe

run majority voting to create ensemble

output:

number of correct predictions for each model

accuracy of each model


# data

images are from testset of the imagenet challenge 2012 

http://www.image-net.org/challenges/LSVRC/2012/

pickle file: prediction mix 

contains sample output for networkPredictions run with resnet, vgg16, densenet and the example images
