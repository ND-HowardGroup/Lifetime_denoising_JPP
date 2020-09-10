close all
clear variables
clc
%author:  Varun Mannam
%date: 11th Jan 2020
format long;
images = 50;

addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/original_images_reg/Tau_stack');
%addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/registration');

s1 = 'W800_P200_4mW_Ax1_Tau1_t';
ip =zeros(512,512);
lower_Tau = 0;
upper_Tau = 10e-9;

ipx = zeros(560,560,images);
ipx3 = zeros(560,560,images);
for i=1:images
    s2 = num2str(i);
    s3 = '.tif';
    s4 = strcat(s1,s2);
    s5 = strcat(s4,s3);
    ip = imread(s5);
    ip1 = mat2gray(ip,[lower_Tau,upper_Tau]);
    %ip1 = mat2gray(ip,[0,0.015]);
    ipx(:,:,i) = ip1;
    ip2 =  ipx(:,:,1); %reference frame
    ipx2 = ipx(:,:,i); %moving frame
    
    [optimizer, metric] = imregconfig('monomodal'); 
    %same single sensor we used to capture all images
    %changing the setting to better one
    optimizer.MaximumIterations = 500; 
    optimizer.GradientMagnitudeTolerance = 1e-6;
    optimizer.RelaxationFactor = 0.1;
    optimizer.MinimumStepLength = 1e-6;
    optimizer.MaximumStepLength = 1e-2;

    ipx3(:,:,i) = imregister(ipx2, ip2, 'rigid', optimizer, metric);
    
    %ip2 = ip1*255;
    %s44 = '/Users/varunmannam/Desktop/Fall19/Research_F19/Dec19/2812/large_dataset/mouse_kidney/Tau_pngs/';
    %file_name = strcat(s44,s4,'.png');
    %imwrite(uint8(ip2), file_name);
    output = ipx3(:,:,i);
    save_to_tiff(output,s5);
end

ipx_mean = mean(ipx, 3);
ipx2_mean = mean(ipx3, 3);


% for i=1:images
%     s2 = num2str(i);
%     s3 = '.tif';
%     s4 = strcat(s1,s2);
%     s5 = strcat(s4,s3);
%     ip = imread(s5);
%     ip2 = ipx(:,:,i)*255;
%     ip22 = ipx3(:,:,i)*255;
%     s44 = '/Users/varunmannam/Desktop/Fall19/Research_F19/Dec19/2812/large_dataset/mouse_kidney/Tau_pngs/';
%     file_name = strcat(s44,s4,'.png');
%     imwrite(uint8(ip2), file_name);
%     
%     s444 = '/Users/varunmannam/Desktop/Fall19/Research_F19/Dec19/2812/large_dataset/mouse_kidney/Tau_pngs_reg/';
%     file_name1 = strcat(s444,s4,'.png');
%     imwrite(uint8(ip22), file_name1);
%     
% end