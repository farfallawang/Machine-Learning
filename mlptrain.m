function [ z, w, v, valid_err_rate, train_err_rate ] = mlptrain(train_data, val_data, h, k, c1, c2)

    %h is the number of hidden units
    [row_num, col_num] = size(train_data);
    
    %initialize stepsize, w, v
    stepsize = 0.01;
    alpha = 0.2;
    w = (0.02)*rand(h,c1*c2+1)-0.01 ; % w = m*65
    v = (0.02)*rand(k,h+1)-0.01;   %v = k*(h + 1)
    
   
    true_label_mat = zeros(row_num,k); %turn column label into matrix
    for class = 1:k
        idx = train_data(:,col_num) == class-1;
        true_label_mat(idx,class) = 1;        
    end
    
    prev_delta_w = zeros(h, c1*c2+1);
    prev_delta_v = zeros(k, h+1);
    prev_error = 0;
    for iter = 1:500
        
        %train 
        idx = randperm(row_num);
        train_data = train_data(idx,:);
        true_label_mat = true_label_mat(idx,:);
        
        cur_error = 0;
        for row = 1 : row_num  
            for hidden = 1 : h 
               z(row, hidden) = 1 / (1 + exp(-(w(hidden, :) * [1, train_data(row, 1:c1*c2)]')));
            end
            
            for i = 1 : k
                oh(row, i) = v(i, :) * [1, z(row, :)]';
            end
            
            for i = 1 : k
                y(row,i) = exp(oh(row,i)) / sum(exp(oh(row,:)));  
            end

            label = train_data(row, c1*c2 +1);
            
            for i = 1 : k
                delta_v(i, :) = stepsize * (true_label_mat(row,i) - y(row,i)) * [1, z(row, :)] + alpha * prev_delta_v(i, :);
            end
            
            for hidden = 1 : h   
                delta_w(hidden, :) = stepsize * ((true_label_mat(row, :) - y(row,:))*v(:,hidden+1))*(1 - z(row, hidden))*z(row,hidden) * [1, train_data(row, 1:c1*c2)] + alpha * prev_delta_w(hidden, :);
            end
            
            v = v + delta_v;
            w = w + delta_w;
            
            prev_delta_w = delta_w;
            prev_delta_v = delta_v;
            
            for i = 1 : k
                if label == i - 1
                    cur_error = cur_error - log(y(row,i));
                end
            end
        end
        
        if abs(cur_error - prev_error) < 0.1
            break;
        end

        prev_error = cur_error;
        
    end
    
    [val_row,val_col] = size(val_data);
    for row = 1 : val_row
        for hidden = 1 : h
            valid_z(row, hidden) = 1 / (1 + exp(-(w(hidden, :) * [1, val_data(row, 1 : c1 * c2)]')));
        end
        for i = 1 : k
            valid_y(row,i) = v(i, :) * [1, valid_z(row, :)]';
        end
    end
    [~, valid_predict_label] = max(valid_y,[],2);
    valid_predict_label = valid_predict_label - ones(val_row, 1);
    valid_err_rate = sum(valid_predict_label ~= val_data(:,val_col))/length(val_data(:,val_col));
    
    for row = 1 : row_num
        for hidden = 1 : h
            z(row, hidden) = 1 / (1 + exp(-(w(hidden, :) * [1, train_data(row, 1:c1*c2)]')));
        end
        for i = 1 : k
            y(row,i) = v(i, :) * [1, z(row, :)]';
        end
    end
    [~, train_predict_label] = max(y,[],2);
    train_predict_label =  train_predict_label - ones(row, 1);
    train_err_rate = sum(train_predict_label ~= train_data(:,col_num))/length(train_data(:,col_num));
end



