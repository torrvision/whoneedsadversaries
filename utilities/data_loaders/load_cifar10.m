function[data] = load_cifar10(data_dir)
% Loosely adapted from sample code in matconvnet's examples folder.
% In our experiments on CIFAR, we have confined data preprocessing to
% subtraction of the (training) dataset mean, forgoing e.g. contrast
% normalisation and data whitening options. While these options can be used
% to get more performance from the classifier, they are not material to the
% point of the paper, and simpler preprocessing means more direct visual
% interpretation of results.
    files{1} = fullfile(data_dir, 'data_batch_1.mat');
    files{2} = fullfile(data_dir, 'data_batch_2.mat');
    files{3} = fullfile(data_dir, 'data_batch_3.mat');
    files{4} = fullfile(data_dir, 'data_batch_4.mat');
    files{5} = fullfile(data_dir, 'data_batch_5.mat');
    files{6} = fullfile(data_dir, 'test_batch.mat');
    im_data = cell(1, numel(files));
    labels = cell(1, numel(files));
    sets = cell(1, numel(files));
    file_set = uint8([ones(1,5),3]);
    for fi = 1:numel(files)
      fd = load(files{fi}) ;
      im_data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
      labels{fi} = fd.labels' + 1; % using 1-based indexing
      sets{fi} = repmat(file_set(fi), size(labels{fi}));
    end

    set = cat(2, sets{:});
    im_data = single(cat(4, im_data{:}));
    labels = single(cat(2,labels{:}));
    
    % (training set) mean subtraction:
    dataMean = mean(im_data(:,:,:,set == 1), 4);
    im_data = bsxfun(@minus, im_data, dataMean);

    clNames = load(fullfile(data_dir, 'batches.meta.mat'));
    
    data.mean = dataMean;
    data.train_images = single(im_data(:,:,:,set==1));
    data.test_images = single(im_data(:,:,:,set==3));
    data.train_labels = labels(1,set==1);    
    data.test_labels = labels(1,set==3); 
    data.image_size = [32 32 3];
    data.labels_limit = 10; % number of classes
    data.meta.sets = {'train','val','test'};
    data.meta.classes = clNames.label_names;
    data.getDataset = set_getDataset();
end


function fn = set_getDataset()
    fn = @(dataset,batch_ids) getDataset_process(dataset, batch_ids);
end


function output = getDataset_process(dataset, batch_ids)
    % For CIFAR10, all images are already loaded.
    output = dataset(:,:,:,batch_ids);
end
