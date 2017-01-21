function [reconstructed] = myPCA(data)
% The fuction takes in a dataset and it first plots the ?rst two eigenvectors 
% to visualize the eigen faces of the dataset 
% It then use the ?rst d principle components to approximate the ?rst image of the dataset 

%center training and testing data
[rowNum,colNum] = size(data);
dataMean = mean(data ,1);
centeredData = data - repmat(dataMean,rowNum,1);

%find eigenvectors and eigenvalues:
[PC,V] = eig(cov(centeredData));
[~,idx] = sort(diag(V),'descend'); %sort variances in decreasing order
PC = PC(:,idx);
%visualize the first two eigenvectors
imagesc(reshape(PC(:,1),60,64));
figure;
imagesc(reshape(PC(:,2),60,64));

%reconstruct using PCA
reconstructedMatrices = cell(1,3);
d = [10; 50; 100];
for idx = 1:length(d)
    %projected = centeredData*PC(:,1:d) ; %156*3840*(3840*10) 
    reconstructed = centeredData*PC(:,1:d(idx))*PC(:,1:d(idx)).';  %(156*10)*(10*3840) = 156*3840 Z = X*V Z*V^T = X 
    reconstructedMatrices{idx} = reconstructed + repmat(dataMean,rowNum,1); %add back mean
    figure;imagesc(reshape(reconstructedMatrices{idx}(1,:),60,64));
end

end


