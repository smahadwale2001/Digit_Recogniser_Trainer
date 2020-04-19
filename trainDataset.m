function [nn_params] = trainData(fxname,fyname,h_size,t_set,opt,lamb_arr)

%----------Initialization-----------
close all;clc;
if (~exist('lamb_arr','var'))
   lamb_arr = [0,0.01,0.03,0.1,0.3,1,3,10,30] 
   fprintf("\nThe Values of Lambda are not Mentioned so Selecting Lambda Values Automatically as Follow\n");
end

if ~exist('t_set','var')
    t_set = 80;
    fprintf("\nSelecting t_set = %f : \n",t_set); 
end
if ~exist('fxname','dir')
    fxname = "TrainExample\train-images.idx3-ubyte"
end

if ~exist('fyname','dir')
    fyname = "TrainExample\train-labels.idx1-ubyte"
end

if ~exist('opt','var') 
    opt(1) = optimset('MaxIter',100);
    opt(2) = optimset('MaxIter',250);
end
%-----------------------------------
loadIF;
Theta1 = 0;
Theta2 = 0;

%==================Loading=================

[X,Y] = loadMNIST(fxname,fyname);
Training_Set_Data_Size = size(X);
Label_Set_Data_Size = size(Y);
t_set = uint64(size(X,1)*t_set/100)
if ~exist('h_size','var')
   h_size = uint64(size(X',1)/8);
   fprintf("As You have not mentioned the size of Hidden Layer , Selecting Hidden Layer of %d", h_size);
end
input_layer_size = size(X',1)        % 50 x 50 img -> Reshaped into 2500 features
hidden_layer_size = double(h_size);        % 150 Hidden Units
num_labels = 10;                % Number Labels 0 - 9 i.e. 10 , Here 10 is labeles for 0
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%===========================================
%-------------Finding Thetas--------------




%-----------------------------------------
%----------Finding Lambda-----------------
lambda = 0;
[J,grad] = costFN(initial_nn_params, input_layer_size, hidden_layer_size,num_labels, X, Y, lambda);
fprintf("Initial Cost is %f\nProgram Paused , Please Enter to Continue\n",J);
pause();
i=1;
for i=1:length(lamb_arr)
    lambda = lamb_arr(i);
    fprintf("Starting Minimization for lambda = %f\n",lambda);
    pause(1);
    costFunction = @(p) costFN(p,input_layer_size,hidden_layer_size,num_labels, X(1:t_set,:), Y(:,1:t_set), lambda);
    [nn_params , J] = fmincg(costFunction, initial_nn_params, opt(1));
    [J,grad] = costFN(nn_params, input_layer_size, hidden_layer_size,num_labels, X, Y, lambda);
    e = errorCompute(nn_params,input_layer_size,hidden_layer_size,num_labels,X,Y,lambda);
    J_arr(i,:) = [e , sqrt(J*J) , lambda];
    fprintf("\nCost is %f\t\tError is : %f\n\n",J,e);
    if(i>1)
        if(J_arr(i,1) > J_arr(i-1))
            break
        end
    end
end

sorted_J = sortrows(J_arr);
lambda_final = sorted_J(1,3);
costFunction = @(p) costFN(p,input_layer_size,hidden_layer_size,num_labels, X, Y, lambda_final);
[nn_params, J] = fmincg(costFunction, nn_params, opt(2));
fprintf("Combination Found out to be Cost : %f\tLambda : %f with error of %f",sorted_J(1,2),sorted_J(1,3),sorted_J(1,1));
%fprintf("Program Paused , Please Enter to Continue");
%pause;
lambda = lambda_final
filename = input("\nEnter Name of File to Save All Parameters : ",'s');
    save(filename,'nn_params','input_layer_size','hidden_layer_size','num_labels','lambda');
end
