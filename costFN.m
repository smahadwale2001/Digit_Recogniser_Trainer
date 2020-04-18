function [J,grad,h_theta] = costFN(nn_params,input_layer_size,hidden_layer_size,num_labels,X,Y,lambda)
J = 0;
grad = 0;
result = 0;

divide = hidden_layer_size*(input_layer_size+1);
Theta1 = reshape(nn_params(1:divide),hidden_layer_size,(input_layer_size+1));

Theta2 = reshape(nn_params((1+divide):(divide+(hidden_layer_size+1)*num_labels)),num_labels,(hidden_layer_size+1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Add more constant feature i.e. 1


X = [ones(m,1) , X];

a1 = X';                        % X -> [100 x 2501]          a1 -> [2501 x 100] 
z2 = Theta1 * a1;               % Theta1 -> [150 x 2501]     z2 -> [150 x 100]
a2 = sigmoid(z2)';              % a2 -> [100 x 150]

a2 = [ones(m,1) , a2];          % a2 -> [100 x 151]
z3 = Theta2 * a2';              % Theta2 -> [10 x 151]       z3 -> [10 x 100]
h_theta = sigmoid(z3);          % output hypothesis -> [10 x 100]
if(Y==0)
    return;
end
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

J = -sum(sum(Y.*log(h_theta) + (1-Y).*log(1-h_theta))) / m + lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

%---------------------------------------------------------------------------------------------------------
%========================================= BACK PROPAGATION ==============================================
%---------------------------------------------------------------------------------------------------------

for i = 1:m
    a1 = X(i,:)';               % a1 -> [2501 x 1]
    z2 = Theta1 * a1;           % z2 -> [150 x 1]
    a2 = sigmoid(z2);           % a2 -> [150 x 1]
  
    a2 = [1;a2];                 % a2 -> [151 x 1]
    z3 = Theta2 * a2;           % z3 -> [10 x 1]
    a3 = sigmoid(z3);           % a3 -> [10 x 1]
   
    delta3 = a3 - Y(:,i);       % delta3 -> [10 x 1]
    z2 = [1;z2];                % z2 -> [151 x 1]
    delta2 = (Theta2'*delta3).*sigmoidGradient(z2);
    delta2 = delta2(2:end);
    Theta2_grad = (Theta2_grad + delta3*a2');
    Theta1_grad = (Theta1_grad + delta2*a1');
end

%---------------------------------------------------------------------------------------------------------

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end