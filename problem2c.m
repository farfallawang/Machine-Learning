function [ ] = problem2c(train_data, val_data, test_data, h, k, c1, c2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes her
   
    train_and_valid = [train_data; val_data];
    Z_train_and_valid = mlptest(train_data, val_data,train_and_valid, h, k, c1, c2);
    [coeff,~,~] = pca(Z_train_and_valid);
    
    Z_test = mlptest(train_data, val_data, test_data, h, k, c1, c2);
    score = Z_test*coeff(:,1:3);
   
    [~, test_label] =  max(Z_test,[],2) ;
    test_label = test_label - ones(size(test_data,1),1);
    
    min_x = min(score(:,1));
    min_y = min(score(:,2));
    min_z = min(score(:,3));
    max_x = max(score(:,1));
    max_y = max(score(:,2));
    max_z = max(score(:,3));
    
    figure;
    for idx = 1 : 10
        check = test_label == (idx -1);
        scatter(score(check, 1), score(check, 2), 8, 'filled');
        str = sprintf('%d', idx-1);
        text(score(check, 1), score(check, 2), str);
        hold on;
    end
    
    hold off;
    figure;
    for idx = 1 : 10
        check = test_label == (idx -1);
        scatter3(score(check, 1), score(check, 2), score(check, 3),'filled');
        axis([min_x max_x min_y max_y min_z max_z]);
        str = sprintf('%d', idx-1);
        text(score(check, 1), score(check, 2),score(check, 2),str);
        hold on;
    end

end

