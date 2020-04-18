function [rz] = sigmoidGradient(z)
rz = zeros(size(z));
rz = sigmoid(z).*(1 - sigmoid(z));
end