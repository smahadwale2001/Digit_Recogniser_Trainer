function error = errorCompute(nn_param,ip_layer,hd_layer,num_labels,X,Y,lambda)
error =0;
counter = 0;
for i=1:size(X)
    counter=counter+1;
    [~,~,result] = costFN(nn_param,ip_layer,hd_layer,num_labels,X(i,:),0,lambda);
    if(predictNo(result) ~= predictNo(Y(:,i)))
        error = error+1;
    end
end
error = 100*error/counter;
end