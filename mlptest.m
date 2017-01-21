function[ z, test_err ] = mlptest(train_data, val_data, test_data, h, k, c1, c2)
    
    test_label = test_data(:, c1 * c2 + 1);
    test_row = size(test_label,1);
    
    [~, w, v, ~, ~] = mlptrain(train_data, val_data, h, k, c1, c2);
    
    for t = 1 : test_row
        for h = 1 : h
            z(t, h) = 1 / (1 + exp(-(w(h, :) * [1, test_data(t, 1: c1 * c2)]')));
        end
        for i = 1 : k
            y(t,i) = v(i, :) * [1, z(t, :)]';
        end
    end
    
    [~, pre_label] = max(y,[],2);
    test_err = sum(pre_label-1 ~= test_label)/test_row;

end

