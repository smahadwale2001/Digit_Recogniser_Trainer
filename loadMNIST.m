function [X,Y] = loadMNIST(imgName,labelName)
X = loadMNISTImages(imgName);
y = loadMNISTLabels(labelName);
for i=1:size(y)
    if(y(i) == 0)
        y(i) = 10;
    end
    Y(:,i) = zeros(10,1);
    Y(y(i),i) = 1; 
end
X = X';
end