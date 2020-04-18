function h_theta = sigmoid(z)
%This function is used to get Sigmoid Value of data
h_theta = 1./(1+exp(-z));
end