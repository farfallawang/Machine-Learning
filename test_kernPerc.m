function [ test_err ] = test_kernPerc( train_data, test_data, sigma)

    [train_row,train_col] = size(train_data);
    [test_row,test_col] = size(test_data);
        
    train_x = [ones(train_row,1), train_data(:,1:train_col-1)];
    train_y = train_data(:,train_col); 
    test_x = [ones(test_row,1), test_data(:,1:test_col-1)];
    test_y = test_data(:,test_col);
    
    %Calculate test kernel
    kernel = zeros(test_row, train_row);
    for row = 1: test_row
        dist = repmat(test_x(row,:),train_row,1) - train_x;
        kernel(row,:) = exp(-sum(dist.^2,2)'/(2*sigma)); %N*N 
    end

    %alpha
    [alpha, ~] = kernPerc(train_data, sigma); %train_row * 1
    
    %Predict
    predict = zeros(test_row, 1);
    for row = 1:test_row
        predict(row) = sign(kernel(row,:)*(train_y.*alpha));  %sign
    end
    
    test_err = sum(test_y ~= predict)/test_row;
        
end

