function [ ] = EMG_revised(filename,k)

%Sometimes it fails due to bad kmeans initialization but if you try more
%initialization it will work

[img,cmap] = imread(filename); 
img_rgb = ind2rgb(img,cmap); 
img_double = im2double(img_rgb);
reshaped_img = reshape(img_double,[],3);
[m,n] = size(reshaped_img);

% Initialize mu, sigma, pi
[cluster_label,mu] = kmeans(reshaped_img,k,'MaxIter',3);
sigma = [ ];
pi = ones(1,k);
for cluster = 1 : k
    sigma{cluster} = cov(reshaped_img(cluster_label == cluster,:));
    pi(cluster) = sum(cluster_label == cluster)/m; %pi = [1 * k]
end

for iter = 1:500
    fprintf('  EM Iteration %d\n', iter);
    
    %E step: Update gamma function
    pdf = zeros(m, k); % pdf [m * k]
    for cluster = 1 : k
        pdf(:, cluster) = mvnpdf(reshaped_img,mu(cluster,:),sigma{cluster});
    end   
    pdf_weighted = bsxfun(@times, pdf, pi);    % pdf_weighted [m * k]  
    gamma = bsxfun(@rdivide, pdf_weighted, sum(pdf_weighted, 2)); % gamma [m * k]
    
    %Complete likelihood after E
    for i = 1 : k
        tmp = log(mvnpdf(reshaped_img, mu(i,:),sigma{i}));
        tmp(tmp == -inf) = 0;
        log_likelihood(i) = gamma(:,i)'*(log(pi(i)) + tmp); 
    end
   
    complete_likelihood_E = sum(log_likelihood, 2);
    complete_likelihood(iter)= complete_likelihood_E;
    
    %M step: Update mu, sigma, and pi
    pi = mean(gamma, 1); % pi = 1*k
    Ni = sum(gamma,1)'; %Ni = k*1
    
    %Complet likelihood after M
    complikelihood_M = sum(log_likelihood, 2);
    complete_likelihood(iter)= complikelihood_M;
    
    % Updating mu
    prevMu = mu;
    mu = gamma'*reshaped_img; 
    mu = bsxfun(@rdivide,mu,Ni);

    %Updating sigma 
    for cluster = 1 : k
        sigma_k = zeros(n, n);
        %Xm = bsxfun(@minus, reshape_img, mu(cluster, :));       
        for row = 1 : m
            xm = reshaped_img(row,:)-mu(cluster,:);
            sigma_k = sigma_k + gamma(row,cluster).*(xm'*xm);
            %sigma_k = sigma_k + (gamma(i, cluster) .* (Xm(i, :)' * Xm(i, :)));
        end        
        sigma{cluster} = sigma_k ./ Ni(cluster);
    end
   
    % Check for convergence.
    if(abs(sum(sum(mu - prevMu))) <= 0.0000001)
        break;
    end
    
end


% Relabel and compress the image
% new_cluster_label = zeros(m,1);
pr = gamma'.*(repmat(pi',1,m));
[~,new_cluster_label] = max(pr, [], 1);

new_img = ones(m,n);
for cluster = 1:k
    idx = (new_cluster_label==cluster);
    new_img(idx,:) = repmat(mu(cluster,:),sum(idx),1);    
end

figure;
new_img = reshape(new_img,size(img_rgb,1),size(img_rgb,2),size(img_rgb,3))  ;
image(new_img);
hold on;

%print complete_likelihood plot
X = 1 : 1 :iter+0.5;
likelihood_E = 1 : 2 : iter;
likelihood_M = 2 : 2 : iter;
figure;
hold on;
scatter(X(likelihood_E), complete_likelihood(likelihood_E), 5, 'r');
scatter(X(likelihood_M), complete_likelihood(likelihood_M), 5, 'b');
hold off



end

