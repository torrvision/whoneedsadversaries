function [result] = test_perf(data, net, gpu, getDataset, add_image)
% A convenience function which uses predict_matconvnet to evaluate a net on
% a dataset, returning the train and test accuracies, and predicted labels.
% - data: dataset, as loaded by a function in data_loaders
% - net: matconvnet (simpleNN) net with softmax layer removed
% - gpu: logical to specify whether to run on GPU (assumes data and net come
%       in on CPU; they will be returned on exit)
% - getDataset: (required) function for accessing images from a collection:
%       defined by the loader used to load the set originally
% - add_image: optional additive universal perturbation to apply to all
%       images fed to the net (scalar 0 works to specify no perturbation)
    train_images = data.train_images;
    test_images = data.test_images;
    train_labels = data.train_labels;
    test_labels = data.test_labels;
    num_cl = length(data.meta.classes);   
    
    batchsize = 200; % If you're having issues with running out of GPU memory, try reducing this.
    conf_flag = 0;    
    
    train_pred = predict_matconvnet(train_images, net, batchsize, gpu, ...
        num_cl, conf_flag, getDataset, add_image);
    train_set_results = (train_pred' == train_labels);    
    test_pred = predict_matconvnet(test_images, net, batchsize, gpu, ...
        num_cl, conf_flag, getDataset, add_image);
    test_set_results = (test_pred' == test_labels);        
    
    train_accuracy = sum(train_set_results) / length(train_set_results);
    test_accuracy = sum(test_set_results) / length(test_set_results);
    
    result.acc = [train_accuracy test_accuracy];    
    result.train_labels = train_pred;
    result.test_labels = test_pred;
    
