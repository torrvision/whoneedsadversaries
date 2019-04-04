function data = load_mnist(data_dir)
% Loads MNIST using helper functions found here:
% http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip
% (The versions used in this project have been included in this repo for
% convenience.)
    % Each image is a column in the matrix of length 784, intended to be
    % resized into a 28x28 square. It's hardcoded here:
    train_images = loadMNISTImages(fullfile(data_dir, 'train-images-idx3-ubyte'));
    test_images = loadMNISTImages(fullfile(data_dir, 't10k-images-idx3-ubyte'));
    
    % mean subtraction:
    mean_train_image = mean(train_images, 2);
    normalised_train_images = train_images - repmat(mean_train_image, [1 size(train_images,2)]);
    normalised_test_images = test_images - repmat(mean_train_image, [1 size(test_images,2)]);
    
    data.mean = mean_train_image;        
    % Transposing to row vectors to match the convention being used in the
    % CIFAR loader, and shifting labels into the range 1-10:
    data.train_labels = (loadMNISTLabels(fullfile(data_dir, 'train-labels-idx1-ubyte'))+1)';
    data.test_labels = (loadMNISTLabels(fullfile(data_dir, 't10k-labels-idx1-ubyte'))+1)';
    data.image_size = [28 28 1];
    data.labels_limit = 10; % number of classes
    data.train_images = reshape(normalised_train_images,[data.image_size, size(train_images,2)]);
    data.test_images = reshape(normalised_test_images,[data.image_size, size(test_images,2)]);
    
    data.meta.classes = {'zero'; 'one'; 'two'; 'three'; 'four'; 'five'; 'six'; 'seven'; 'eight'; 'nine'};
    data.meta.name = 'MNIST';
    data.getDataset = set_getDataset();
end


function fn = set_getDataset()
    fn = @(dataset,batch_ids) getDataset_process(dataset, batch_ids);
end


function output = getDataset_process(dataset, batch_ids)
    % For MNIST, the images are already loaded.
    % There is a singleton 3rd dimension to establish consistency with the
    % colour datasets: the 4th index selects the image.
    output = dataset(:,:,:,batch_ids);
end
