global plan
addpath(genpath('.'));
% json = ParseJSON('plans/mnist_simple.txt');
json = ParseJSON('plans/mnist_simple_dropout.txt');
% json = ParseJSON('plans/mnist_simple_dropin.txt');
Plan(json);

Run();


% Fully connected without dropout (768-800-800-10):
% ~180 errors after 50 epochs with final lr = 0.01
% 
% Fully connected with 0.5 dropout (768-800-800-10):
% ~147 error after 73 epochs with final lr = 0.01
% 
% Fully connected with 0.2 dropin, patial bp (768-800-800-10):
% ~145 error after 450 epochs with final lr = 0.01
% 
% Fully connected with 0.1 dropin, partial bp (768-800-800-10):
% ~158 error after 89 epoch with final lr = 0.01
% 
% Fully connected with 0.3 dropin, partial bp (768-800-800-10):
% ~226 error after 90 epoch with final lr = 0.1
% 
% Fully connected with 0.05 dropin, 0.5 dropout, partial bp (768-800-800-10):
% ~133 error with lr = 0.1 and ~140 with final lr = 0.01
% 
% Fully connected with 0.1 dropin, 0.5 dropout, partial bp (768-800-800-10):
% ~139 error with lr = 0.1
% 


% XXXXX : Share it with Christian