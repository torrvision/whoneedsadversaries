function [data] = load_imagenet(path, net, im_size)
% A "loader" which actually loads the names of the images, rather than the
% images themselves, to account for ImageNet's unwieldy size. The
% getDataset function supplied with the returned structure data can then be
% used to load the actual images in batches when actual processing is being
% done. (Note that this accepts a resizing parameter for facilitating
% certain experiments, as described in the paper.)
    load(strcat(path,'/imdb.mat'));
    data.train_images = fullfile(imageDir, images.name(images.set==1));
    data.train_labels = images.label(images.set==1);
    data.test_images = fullfile(imageDir, images.name(images.set==2));
    data.test_labels = images.label(images.set==2);
    data.image_size = net.meta.inputSize(1:3);
    data.labels_limit = size(classes.name, 2);
    data.meta.sets = {'train','val','test'};
    data.meta.classes = classes.description;
    data.getDataset = set_getDataset(net.meta, im_size);
    
end


function fn = set_getDataset(meta, im_size)

    if numel(meta.normalization.averageImage) == 3
      mu = double(meta.normalization.averageImage(:)) ;
    else
      mu = imresize(single(meta.normalization.averageImage), ...
                    meta.normalization.imageSize(1:2)) ;
    end
    test_opts = struct(...
      'useGpu', false, ...
      'numThreads', 4, ...
      'imageSize',  meta.normalization.imageSize(1:2), ...
      'cropSize', meta.normalization.cropSize, ...
      'subtractAverage', mu) ;
    fn = @(dataset,batch_ids) getDataset_process(test_opts, im_size, ...
        dataset, batch_ids);
    
end


function output = getDataset_process(opts, im_size, dataset, batch_ids)

    if ismatrix(dataset)
        % (The images need to be loaded and preprocessed on the fly.)
        images = dataset(batch_ids) ;
        output = getImageBatch(images, opts, 'prefetch', nargout == 0) ;    
    else
        disp('expecting the images to be pre-loaded here');
        % (The images are already loaded.)
        output = dataset(:,:,:,batch_ids);
    end

    % The below expects the image to be a *square* matrix:
    if im_size == opts.imageSize(1)
    % ... then don't do anything.
    else
        orig_size = opts.imageSize(1);
        for i=1:size(output,4)
            % Get the image, downsample it, and upsample it back again.
            output(:,:,:,i) = imresize(imresize(output(:,:,:,i), ...
                [im_size, im_size]), [orig_size, orig_size]);
        end
    end

end

% SJ/NL: The following subroutine was copied in from MatConvNet:
% -------------------------------------------------------------------------
function data = getImageBatch(imagePaths, varargin)
% -------------------------------------------------------------------------
% GETIMAGEBATCH  Load and jitter a batch of images

opts.useGpu = false ;
opts.prefetch = false ;
opts.numThreads = 1 ;

opts.imageSize = [227, 227] ;
opts.cropSize = 227 / 256 ;
opts.keepAspect = true ;
opts.subtractAverage = [] ;

opts.jitterFlip = false ;
opts.jitterLocation = false ;
opts.jitterAspect = 1 ;
opts.jitterScale = 1 ;
opts.jitterBrightness = 0 ;
opts.jitterContrast = 0 ;
opts.jitterSaturation = 0 ;

opts = vl_argparse(opts, varargin);

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Pack', ...
           'Interpolation', 'bicubic', ...
           'Resize', opts.imageSize(1:2), ...
           'CropSize', opts.cropSize * opts.jitterScale, ...
           'CropAnisotropy', opts.jitterAspect, ...
           'Brightness', opts.jitterBrightness, ...
           'Contrast', opts.jitterContrast, ...
           'Saturation', opts.jitterSaturation} ;

if ~opts.keepAspect
  args{end+1} = {'CropAnisotropy', 0} ;
end

if opts.jitterFlip
  args{end+1} = {'Flip'} ;
end

if opts.jitterLocation
  args{end+1} = {'CropLocation', 'random'} ;
else
  args{end+1} = {'CropLocation', 'center'} ;
end

if opts.useGpu
  args{end+1} = {'Gpu'} ;
end

if ~isempty(opts.subtractAverage)
  args{end+1} = {'SubtractAverage', opts.subtractAverage} ;
end

args = horzcat(args{:}) ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
else
  data = vl_imreadjpeg(args{:}) ;
  data = data{1} ;
end
end
