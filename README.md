# Facial_Recognition_Emotions_CNN_MLP
## Link for dataset

 https://drive.google.com/file/d/1Mu45czJuWlgW9zXefxXn4TLy2KPYsI2Z/view?usp=sharing


## Remarque : 
we can use this update from (DEBBAGH Abdelkader Nadir) :


"ConvX : représente une convolution avec X filtres de taille 3x3 suivie d'un ReLU 
BN : représente un Batch Normalization 
Dropout : représente un dropout avec proba 0.5
MP : représente un maxpool (2,2)
DenseX : représente une couche fully connected avec X neurones suivie d'un ReLU

Architecture (les lignes se suivent):
Conv3, BN, Conv16, BN, Conv32, BN, Conv32, BN, MP, 
Conv64, BN, Conv64, BN, MP                                          
Conv128, BN, Conv128, BN, MP,                                     
Conv256, BN, Conv256, BN, Conv256, BN, MP           
Flatten, Dense1024, BN, Dropout, Dense512, BN, Dropout, Dense7

J'ai utilisé rmsprop, avec de la Data Augmentation : rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True"
