load optdigits_train.txt
load optdigits_valid.txt
load optdigits_test.txt

%====================================================
disp('*********************Question 2a***********************');
c1 = 8;
c2 = 8;
k = 10;
h_vals = [3 6 9 12 15 18];
valid_err_lst = zeros(length(h_vals),1);
train_err_lst = zeros(length(h_vals),1);
for i = 1:length(h_vals)
    h = h_vals(i);
    [z, w, v, train_error_rate, valid_error_rate] = mlptrain(optdigits_train, optdigits_valid, h, k, c1, c2);
    valid_err_lst(i) =  valid_error_rate;
    train_err_lst(i) = train_error_rate;
end

[~, idx] = sort(valid_err_lst);
best_m = h_vals(idx(1));


%====================================================
disp('**********************Question 2b**********************');
problem2b(optdigits_train, optdigits_valid, best_m, k, c1, c2);


%====================================================
disp('**********************Question 2c**********************');
problem2c(optdigits_train, optdigits_valid, optdigits_test, best_m, k, c1, c2);
