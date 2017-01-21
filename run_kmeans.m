function [] = run_kmeans(filename, k)

[img cmap] = imread(filename);
img_rgb = ind2rgb(img, cmap);
img_double = im2double(img_rgb);

reshaped_img = reshape(img_double, [], 3);
[m, ~] = size(reshaped_img);

[idx, mu] = kmeans(reshaped_img, k);

compressed_img = zeros(m, 3);
for i = 1 : k
    cluster_labels = (idx == i);
    compressed_img(cluster_labels, :) = repmat(mu(i,:), sum(cluster_labels), 1);
end

figure;
new_image = reshape(compressed_img,size(img_rgb,1),size(img_rgb,2),size(img_rgb,3));
image(new_image);
hold on;