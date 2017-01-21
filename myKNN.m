function [predict,err] = myKNN(trainingData,testData,knn)
% predict test data based on the classes of its k nearest neighbours in
% the training set

[trainRowNum, trainColNum] = size(trainingData);
[testRowNum, testColNum] = size(testData);
trainingDataWithoutLabels = trainingData(:,1:trainColNum-1);
testingDataWithoutLabels = testData(:,1:testColNum-1);

predict = zeros(1,testRowNum).';
for i = 1:testRowNum
     dist = zeros(1,trainRowNum);
%     for j = 1:trainRowNum
%         tmp = trainingData(j,:) - testingData(i,:);
%         dist(j) = sqrt(sum(tmp(:,:).^2));
%     end    
    dist = trainingDataWithoutLabels - repmat(testingDataWithoutLabels(i,:),trainRowNum,1); 
    dist = sum(dist(:,:).^2,2);
    [~, I] = sort(dist);
    idx = I(1:knn); %knn index
    knn_labels = trainingData(idx,trainColNum); %the trainColNum misses a label column
    predict(i) = mode(knn_labels);
end 

err = sum(predict ~= testData(:,testColNum))/testRowNum;

display('Error rate');
display(err);

end

