function [no] = predData(f,w_name,opt)
loadIF;
no=0;
load(w_name);
    if(opt == 1)
        X = f;
    elseif(opt == 2)
        x = imread(f);
        [m n] = size(x);
        X = reshape(x/255,m*n,1);
    else
        fprintf("");
        return;
    end
[~,~,result] = costFN(nn_params,input_layer_size,hidden_layer_size,num_labels,X,0,lambda_final);
no = predictNo(result);
fprintf("Predicted Number is %f",no);
end
