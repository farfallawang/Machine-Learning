function [ ] = drawDecisionBoundaries(data,knn,testData)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


minX = min(data(:,1));
maxX = max(data(:,1));
minY = min(data(:,2));
maxY = max(data(:,2));
x =  minX : (maxX - minX)/99: maxX; 
y =  minY : (maxY - minY)/99: maxY;
[X, Y] = meshgrid(x, y);
X = reshape(X, [(100)^2 1]);
Y = reshape(Y, [(100)^2 1]);
coordinate = [X Y];
coordinate = cat(2, coordinate, zeros(100^2, 1));

if (knn <= 5)
    [labels, ~] = myKNN(data, coordinate, knn);
    coordinate_labels = reshape(labels, [size(x,2), size(y,2)]);
    imagesc(x, y, coordinate_labels);
    hold on;
    set(gca, 'ydir', 'normal');
    colormap([
        0 1 1; 
        0 0 1; 
        1 0 0;
        0 1 0;
        1 0 1;
        1 1 0;
        1 1 1;
        0.2, 0.7, 0.6; 
        0.2, 0.1, 0.5; 0.1, 0.5, 0.8]);

    for index = 1 : 10
        check_class(:, index) = testData(:, size(testData,2)) == (index -1);
    end

    for index = 1 : 10
        scatter(data(check_class(:, index), 1), data(check_class(:, index), 2), 12, 'filled');
        str = sprintf('%d', index);
        text(data(check_class(:, index), 1), data( check_class(:, index), 2), str)
    end
end



end

