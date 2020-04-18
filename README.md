# Digit_Recogniser_Trainer
This Project consist of Two Layer Neural Network with One Hidden Layer and you can Train your Dataset without any making any code. 

i. How to Train Data : 
I wll suggest you to go for http://yann.lecun.com/exdb/mnist/ this dataset.
It consist of Training and Cross Validation Datasets.

To Train Data use trainDataset :
usage:
trainDataset("dataset directory","label directory",hidden_layer_size,t_set,opt,lamb_arr)

Here,
t_set -> For cross Validation and Error Optimization you have to give percentage of Training Dataset for Training
Ex:
If I have 10000 training set ,if t_set = 90 then it will optimise lambda according to dataset of 9000 with minimization of error.

opt ->
You have to Pass opt struct using optimset.
Ex:
opt(1) = optimset('MaxIter',50);
opt(2) = optimset('MaxIter',100);
trainDataset([],[],100,90,opt);
Remember , opt(2) will optimize at Last , I will suggest to set MaxIter at minimum 100 for this stage.

lamb_arr ->
Send array for Different arrays you want. If you don't send array it will automatically set this to
lamb_arr = [0.01,0.03,0.1,0.3,1,3,10,30]

I have already found weights for mentioned dataset.
