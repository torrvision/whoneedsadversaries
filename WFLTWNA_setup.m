%%
% Run this file once to set the project environment up.
% The user must supply the two path strings below.

MATCONVNET_PATH = ''; % <- path to the top-level matconvnet directory
WFLTWNA_PROJECT_PATH = ''; % <- path to this project's top-level directory

run(fullfile(MATCONVNET_PATH, 'matlab', 'vl_setupnn'));
root = vl_rootnn();
% We use cnn_imagenet_deploy, so we ensure the inclusion of this directory:
addpath(fullfile(root, 'examples', 'imagenet'));

addpath(genpath(WFLTWNA_PROJECT_PATH));

clear;