function [ alpha, err_rate] = kernPerc( train_data, sigma )
     
   [train_row,train_col] = size(train_data);   
    x = [ones(train_row,1), train_data(:,1:train_col-1)]; %x = train_data(:,1:train_col-1);
    y = train_data(:,train_col);
    
    %Calculate kernel
    kernel = zeros(train_row, train_row);
    for row = 1: train_row
        dist = repmat(x(row,:),train_row,1) - x;
        kernel(row,:) = exp(-sum(dist.^2,2)'/(2*sigma)); %N*N 
    end
    
    %Calculate alpha
    alpha = zeros(train_row, 1);
    iter = 0;
    while iter < 500
        for row = 1:train_row
                 if (kernel(row,:)*(y.*alpha)*y(row)) <= 0
                     alpha(row) = alpha(row) + 1;
                 end
        end
    iter = iter + 1;
    end
    
    %Predict
    predict = zeros(train_row, 1);
    for row = 1:train_row
        predict(row) = sign(kernel(row,:)*(y.*alpha));  %sign
    end
    
    err_rate = sum(y ~= predict)/length(y);
    
end

