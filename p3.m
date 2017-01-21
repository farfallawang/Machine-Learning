rng(1)

r = sqrt(rand(100,1));
t = 2*pi*rand(100,1);
data1 = [r.*cos(t), r.*sin(t)];

r2 = sqrt(3*rand(100,1)+2);
t2 = 2*pi*rand(100,1);
data2 = [r2.*cos(t2), r2.*sin(t2)];

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
ezpolar(@(x)1);
ezpolar(@(x)2);
axis equal
hold off

data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;
test_data = [data3, theclass];

sigma = 1;
d = 0.01;
x = min(data2):d:max(data2);
y = min(data2):d:max(data2);
[X,Y] = meshgrid(x,y);
X_reshape = reshape(X,[],1);
Y_reshape = reshape(Y,[],1);
cord = [X_reshape,Y_reshape];

svm_model = fitcsvm(data3,theclass,'KernelScale','auto','Standardize',true,'KernelFunction','RBF','BoxConstraint',1);
[label,~] = predict(svm_model,[X(:) Y(:)]);
figure;
decision_map = reshape(label,[size(x,2) size(y,2)]);
imagesc(x,y,decision_map);
hold on
set(gca,'ydir','normal');
cmap = [1 1 1;
        0 1 0];
colormap(cmap);

% [error] = test_kernPerc(test_data,[cord,ones(size(cord,1),1)],sigma);
% decision_map = reshape(predict,[size(x,2) size(y,2)]);
% figure;
% imagesc(x,y,decision_map);
% hold on
% set(gca,'ydir','normal');
% cmap = [1.0 0.0 0.0;
%         0.0 1.0 0.0];
% colormap(cmap);
% hold off;
% disp(error);
