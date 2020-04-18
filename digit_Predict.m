function digit_Predict(fname,weights)
loadIF;
X = loadMNISTImages(fname);
X=X';
load(weights);
    for i=1:size(X,1)
        img = reshape(X(i,:),28,28);
        [~,~,result] = costFN(nn_params,input_layer_size,hidden_layer_size,num_labels,img,0,lambda_final);
        Pred_Num = predictNo(result);
        imshow(img);
        label = "Predicted Number is "+string(Pred_Num);
        xlabel(label);
        clc;
        loadIF;
        fprintf("Press Key To Continue else press CTRL+C");
        pause;
    end
end