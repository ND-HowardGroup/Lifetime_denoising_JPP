close all
clear variables
clc

%author: Varun Mannam
%date: 12th Jan 2020
%bpae cell G and S images

addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/no_registration/cropped/G_stack');
addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/test_results/G_results/no_reg');
addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/no_registration/cropped/S_stack');
addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/test_results/S_results/no_reg');

addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/original_images_no_reg/cropped/I_stack');
addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/original_images_no_reg/cropped/Tau_stack');
addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/Jan20/1101/BPAE_sample4/original_images_no_reg/cropped');
%addpath('/Users/varunmannam/Desktop/Fall19/Research_F19/Dec19/3112/31stDec2019/BPAE_Sample4/cropped/I_stack');
%addpath('/Users/varunmannam/Desktop/Fall19/Research_F19/Dec19/2812/Keras_inference/I_pngs_reg');
%addpath('/Users/varunmannam/Desktop/Fall19/Research_F19/Dec19/3112/31stDec2019/BPAE_Sample4');

num_images = 50;
font = 14;
gmin=-0.1;
gmax = 1.1;
smin = -0.1;
smax  = 0.6;
mod_freq = 80e6;
hmax = 0.7;
hmin = 0;

x_ind = [1:50]'; 
tau_min = 0e-9;
tau_max = 10e-9;
imsize = 512; %image size in x or y

%input = zeros(512,512);
%denoised = zeros(512,512);
%index = ceil(rand(1)*50);
denoised_g = load('test_nbn_Estimated_result_G_not_registered_bpae_512.mat');
denoised_g = denoised_g.test_nbn_Estimated_result_G_not_registered_bpae_512;
denoised_g = permute(denoised_g,[2 3 1]);

denoised_s = load('test_nbn_Estimated_result_S_not_registered_bpae_512.mat');
denoised_s = denoised_s.test_nbn_Estimated_result_S_not_registered_bpae_512;
denoised_s = permute(denoised_s,[2 3 1]);

input_G = zeros(512,512,num_images);
input_S = zeros(512,512,num_images);
input_tau = zeros(512,512,num_images);
input_intensity = zeros(512,512,num_images);

denoised_G = zeros(512,512,num_images);
denoised_S = zeros(512,512,num_images);
denoised_tau = zeros(512,512,num_images);

target_g = Tiff('Average_G_stack_no_reg_cropped.tif');
target_g = read(target_g); %it is already between 0 to 1
target_g = double(target_g);
%target_g = mat2gray(target_g,[gmin,gmax]);

target_s = Tiff('Average_S_stack_no_reg_cropped.tif');
target_s = read(target_s);%it is already between 0 to 1
target_s = double(target_s);
%target_s = mat2gray(target_s,[smin,smax]);

%pnsr_res_g = zeros(num_images, 2);
%pnsr_res_s = zeros(num_images, 2);
pnsr_res_tau = zeros(num_images, 2);

target_tau = Tiff('AVG_TaU_stack.tif');
target_tau = read(target_tau);
target_tau = double(target_tau); %all ranges
%now limit to tau_min and tau_max
for ii =1:imsize
    for jj=1:imsize
      if target_tau(ii,jj)>tau_max
            target_tau(ii,jj) = tau_max;
      end
      if target_tau(ii,jj)<tau_min
            target_tau(ii,jj) = tau_min;
      end  
    end
end

target_intensity = Tiff('AVG_I_stack.tif');
target_intensity = read(target_intensity);
target_intensity = double(target_intensity); %all ranges

for i =1:num_images
str1 = 'W800_P200_4mW_Ax1_G_t';
str3 = '.tif';
str2 = num2str(i);
%str4 = 'Denoised_reg_mouse_kidney';
str5 = 'W800_P200_4mW_Ax1_S_t';
str6 = 'W800_P200_4mW_Ax1_Tau1_t'; %tau
str7 = 'W800_P200_4mW_Ax1_I_t'; %intensity

file1 = strcat(str1,str2,str3);
%file2 = strcat(str4,str2,str3);
file3 = strcat(str5,str2,str3);
file4 = strcat(str6,str2,str3); %tau file
file5 = strcat(str7,str2,str3); %intensity file

x1 = Tiff(file1);
x2 = Tiff(file3);
x3 = Tiff(file4);
x4 = Tiff(file5);

input_G(:,:,i) = read(x1);
input_G(:,:,i) = mat2gray(input_G(:,:,i),[gmin,gmax]);

%denoised = imread(file2);
input_S(:,:,i) = read(x2);
input_S(:,:,i) = mat2gray(input_S(:,:,i),[smin,smax]);

%lifetime read
input_tau(:,:,i) = read(x3); 
input_tau(:,:,i) = double(input_tau(:,:,i));
%all ranges to tau-min and tau_max
for j=1:imsize
    for k=1:imsize
        if input_tau(j,k,i)>tau_max
            input_tau(j,k,i) = tau_max;
        end
        if input_tau(j,k,i)<tau_min
            input_tau(j,k,i) = tau_min;
        end
    end
end

input_intensity(:,:,i) = read(x4); 
input_intensity(:,:,i) = double(input_intensity(:,:,i));

%denoised results
%G
denoised_G(:,:,i) = denoised_g(:,:,i);
denoised_G(:,:,i) = double(denoised_G(:,:,i)); %to double
denoised_G(:,:,i) = (denoised_G(:,:,i)+0.5); %add 0.5

%S
denoised_S(:,:,i) = denoised_s(:,:,i);
denoised_S(:,:,i) = double(denoised_S(:,:,i)); %to double
denoised_S(:,:,i) = (denoised_S(:,:,i)+0.5); %add 0.5

%pnsr_res_g(i,:) = calculate_PSNR_512(input_G(:,:,i), target_g, denoised_G(:,:,i));
%pnsr_res_s(i,:) = calculate_PSNR_512(input_S(:,:,i), target_s, denoised_S(:,:,i));

%get lifetime directly
denoised_G(:,:,i) = (denoised_G(:,:,i)*1.2) - 0.1; %reverse back to -0.1 to 1.1
denoised_S(:,:,i) = (denoised_S(:,:,i)*0.7) - 0.1; %reverse back to -0.1 to 0.6
denoised_tau(:,:,i) = (denoised_S(:,:,i)./denoised_G(:,:,i))/(2*pi*mod_freq); %all ranges based on G and S images

for j=1:imsize
    for k=1:imsize
        if denoised_tau(j,k,i)>tau_max
            denoised_tau(j,k,i) = tau_max;
        end
        if denoised_tau(j,k,i)<tau_min
            denoised_tau(j,k,i) = tau_min;
        end
    end
end

pnsr_res_tau(i,:) = calculate_lifetime_PSNR_512(input_tau(:,:,i), target_tau, denoised_tau(:,:,i));
end

% figure(1), 
% plot(x_ind, pnsr_res_tau(:,1),'ro','Linewidth',1);
% hold on
% plot(x_ind, pnsr_res_tau(:,2),'b*','Linewidth',1);
% xlabel('Sample index');
% ylabel('PSNR');
% legend('noisy PSNR','denoised PSNR','Location', 'best');
% title('PSNR Comparison');
% set(gca,'FontSize',font);

diff = pnsr_res_tau(:,2) - pnsr_res_tau(:,1);
[max_val,max_ind] = max(diff);

% % customized hsv colormap
m = 1000; % number of colormap bins
map_hue = linspace(hmax,hmin,m)'; % scale of hue
map_saturation = ones(m,1);
map_value = ones(m,1);
hsvmap = [map_hue map_saturation map_value];
rgbmap = hsv2rgb(hsvmap);

%indx = ceil(rand(1)*50);
% figure(2)
% subplot(1,3,1), imagesc(input_tau(:,:,max_ind));
% colormap('gray'), colorbar;
% title('noisy input lifetime')
% set(gca,'FontSize',font);
% subplot(1,3,2), imagesc(denoised_tau(:,:,max_ind))
% colormap('gray'), colorbar;
% title('denoised lifetime')
% set(gca,'FontSize',font);
% subplot(1,3,3), imagesc(target_tau)
% colormap('gray'), colorbar;
% title('target lifetime')
% set(gca,'FontSize',font);


tau_min_d = 1e-9;
tau_max_d = 4e-9;
figure(3)
subplot(1,3,1), imagesc(input_tau(:,:,max_ind),[tau_min_d,tau_max_d]);
colormap('gray'), colorbar;
title('noisy input lifetime')
set(gca,'FontSize',font);
subplot(1,3,2), imagesc(denoised_tau(:,:,max_ind),[tau_min_d,tau_max_d])
colormap('gray'), colorbar;
title('denoised lifetime')
set(gca,'FontSize',font);
subplot(1,3,3), imagesc(target_tau,[tau_min_d,tau_max_d])
colormap('gray'), colorbar;
title('target lifetime')
set(gca,'FontSize',font);


%assign last values
noisy_input_lt = input_tau(:,:,max_ind);
denoised_lt = denoised_tau(:,:,max_ind);
target_lt = target_tau;

%save to .tif file
save_to_tiff(noisy_input_lt, 'noisy_lifetime_image.tif'); 
save_to_tiff(denoised_lt, 'denoised_lifetime_image.tif'); 
save_to_tiff(target_lt, 'target_lifetime_image.tif'); 

%normalise to 0 to 1 from 0 to 10nsec
noisy_input_lt = noisy_input_lt/tau_max; %normalise between 0 to 10nsec
denoised_lt = denoised_lt/tau_max; %normalise between 0 to 10nsec
target_lt = target_lt/tau_max; %normalise between 0 to 10nsec

%select range for RGB image
noisy_input_lt = mat2gray(noisy_input_lt,[0.1,0.4]);
denoised_lt = mat2gray(denoised_lt,[0.1,0.4]);
target_lt = mat2gray(target_lt,[0.1,0.4]);

noisy_input_int = input_intensity(:,:,max_ind);
noisy_input_int = mat2gray(noisy_input_int,[2e-3,5e-3]);

%old code from here onwards
%noisy_input_s = mat2gray(noisy_input_s,[-0.1,0.6]); 
%adjusted brighness manually to get better display

% denoised_g_res = denoised_g(:,:,max_ind );
% denoised_g_res = double(denoised_g_res); %to double
% denoised_g_res = denoised_g_res+0.5;  %add 0.5 here
% denoised_g_res = mat2gray(denoised_g_res,[0.1,0.4]);
% target_lt =  target;
% target_lt = mat2gray(target_lt,[0.1,0.4]);

hsv1 = zeros(512,512,3);
hsv2 = zeros(512,512,3);
hsv3 = zeros(512,512,3);

hsv1(:,:,1) = hmax + (hmin-hmax)*noisy_input_lt;
hsv1(:,:,2) = ones(512,512);
hsv1(:,:,3) = noisy_input_int;
rgb1 = hsv2rgb(hsv1);

hsv2(:,:,1) = hmax + (hmin-hmax)*denoised_lt;
hsv2(:,:,2) = ones(512,512);
hsv2(:,:,3) = noisy_input_int;
rgb2 = hsv2rgb(hsv2);


hsv3(:,:,1) = hmax + (hmin-hmax)*target_lt;
hsv3(:,:,2) = ones(512,512);
hsv3(:,:,3) = noisy_input_int;
rgb3 = hsv2rgb(hsv3);


figure(4),
subplot(1,3,1), imshow(rgb1);
colormap(rgbmap);
title('noisy lifetime')
set(gca,'FontSize',font);
c = colorbar;
c.Label.String = 'Lifetime (s)';
c.Label.FontSize = font;
caxis([tau_min_d tau_max_d])
axis equal tight


subplot(1,3,2), imshow(rgb2);
colormap(rgbmap);
title('denoised lifetime')
set(gca,'FontSize',font);
c = colorbar;
c.Label.String = 'Lifetime (s)';
c.Label.FontSize = font;
caxis([tau_min_d tau_max_d])
axis equal tight


subplot(1,3,3), imshow(rgb3);
colormap(rgbmap);
title('target lifetime')
set(gca,'FontSize',font);
c = colorbar;
c.Label.String = 'Lifetime (s)';
c.Label.FontSize = font;
caxis([tau_min_d tau_max_d])
axis equal tight



%to tif image
save_to_tiff(rgb1, 'noisy_lifetime_rgb_image.tif'); %here intensity is not denoised
save_to_tiff(rgb2, 'denoised_lifetime_rgb_image.tif'); %here intensity is not denoised
save_to_tiff(rgb3, 'target_lifetime_rgb_image.tif'); %here intensity is not denoised

%to png image
imwrite(rgb1, 'noisy_lifetime_rgb_image_png.png'); %here intensity is not denoised
imwrite(rgb2, 'denoised_lifetime_rgb_image_png.png'); %here intensity is not denoised
imwrite(rgb3, 'target_lifetime_rgb_image_png.png');
% 
% 
% display only lifetime in color and with colorbar of 0 to 10nsec
% noisy_input_lt1 = input_lifetime(:,:,max_ind );
% noisy_input_lt1 = noisy_input_lt1*10e-9;
% denoised_lt1 = denoised_lifetime(:,:,max_ind );
% denoised_lt1 = denoised_lt1*10e-9;
% target_lt1 =  target;
% target_lt1 = target_lt1*10e-9;
% 
% figure(3),
% subplot(1,3,1), imshow(noisy_input_lt1);
% title('noisy lifetime')
% c = colorbar;
% c.Label.String = 'Lifetime (s)';
% c.Label.FontSize = font;
% caxis([lifetime_min lifetime_max])
% axis equal tight
% set(gca,'FontSize',font);
% 
% subplot(1,3,2), imshow(denoised_lt1);
% colormap(rgbmap);
% title('denoised lifetime')
% c = colorbar;
% c.Label.String = 'Lifetime (s)';
% c.Label.FontSize = font;
% caxis([lifetime_min lifetime_max])
% axis equal tight
% set(gca,'FontSize',font);
% 
% subplot(1,3,3), imshow(target_lt1);
% colormap(rgbmap);
% title('target lifetime')
% c = colorbar;
% c.Label.String = 'Lifetime (s)';
% c.Label.FontSize = font;
% caxis([lifetime_min lifetime_max])
% axis equal tight
% set(gca,'FontSize',font);
% 
% 
% save_to_tiff(noisy_input_lt1, 'noisy_lifetime_image.tif'); %here intensity is not denoised
% save_to_tiff(denoised_lt1, 'denoised_lifetime_image.tif'); %here intensity is not denoised
% save_to_tiff(target_lt1, 'target_lifetime_image.tif'); %here intensity is not denoised