# Semi-Supervised-Learning

Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision.

Intuitively, the learning problem can be seen as an exam and labeled data as sample problems that the teacher solves for the class as an aid in solving another set of problems. In the transductive setting, these unsolved problems act as exam questions. In the inductive setting, they become practice problems of the sort that will make up the exam.
(source Wikipedia)


Algortihm
  - Split the dataset in 75:25 ration
  - Use 75% data to train the model
  - Augment/transform the rest 25% data make copies (5-10 per image)
  - Then using pretrained model predict the labels for augmented image and validate using true labels.
 
Project structure 
  - data.py ( download the Imagewoof dataset and save it. Read the images and labels using opencv)
  - dataloader.py (split the data and create dataloaders for training and testing)
  - utils.py (helper functions)
  - models.py (initialse resnet18 base model)
  - train.py (Semi supervised learning algorithm)

other files : 
  - requirements.txt (run : python3 -m pip install -r requirements.txt to install all required libraries)
 
I used resnet18 from torchvision.models as teacher model. 
link of dataset "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz" 

The accuracy of the algorithm is 60% for 75 : 25 supervised - unsupervised learning ratio.

