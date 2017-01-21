function [trainPCA, testPCA] = PCA_helper(trainingData,testData,n)
%find the projection matrix  
%project the original data onto the matrix

%center training and testing data
[trainRowNum,trainColNum] = size(trainingData);
[testRowNum,testColNum] = size(testData);
trainingDataWithoutLabels = trainingData(:,1:trainColNum-1);
testingDataWithoutLabels = testData(:,1:testColNum-1);

traingMean = mean(trainingDataWithoutLabels ,1);
centeredTrainingData = trainingDataWithoutLabels - repmat(traingMean,trainRowNum,1);
testingMean = mean(testingDataWithoutLabels,1);
centeredTestingData = testingDataWithoutLabels - repmat(testingMean,testRowNum,1);

%find eigenvectors and eigenvalues:
[PC,V] = eig(cov(centeredTrainingData));
[~,idx] = sort(diag(V),'descend'); %sort variances in decreasing order
PC = PC(:,idx);

trainPCA = centeredTrainingData(:,:)*PC(:,1:n);
trainPCA = [trainPCA, trainingData(:,trainColNum)]; %add label to training data
testPCA = centeredTestingData(:,:)*PC(:,1:n);
testPCA = [testPCA, testData(:,testColNum)]; %add label to testing data

end
