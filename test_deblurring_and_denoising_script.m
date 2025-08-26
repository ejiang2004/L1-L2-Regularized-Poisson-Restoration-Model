% L1/L2 experiment

clear all; clc;
%% Set parameter and peak
delta = 1e-3; %penalty parameter for ADMM
peak = 30;

%% river

% read image
I = imread('river.jpg');
I = double(I);
rng(1234);

% Set image peak and add Poisson noise
Q = max(max(I)) /peak;
I = I / Q;
I(I == 0) = min(min(I(I > 0)));

% apply gaussian blur to image
A=fspecial('gaussian', [10 10], 2);
I_blurry = myconv(I, A);

% u0 = I_blurry;
u0 = imnoise(uint8(I_blurry),'poisson');
u0 = double(u0);

% compute psnr/ssim
noisy_psnr = psnr(u0*Q, I*Q, 255);
noisy_ssim = ssim(uint8(u0*Q), uint8(I*Q));

% set parameters
beta = 0.1;
rho = 1;
iteration = 300;
t1=0;
t2=30;

% L1/L2 denoised
u_denoised = poisson_L1L2_tv_admm(u0, A, beta, rho, t2);

% PSNR + SSIM
denoised_psnr = psnr(u_denoised*Q, I*Q, 255);
denoised_ssim = ssim(uint8(u_denoised*Q), uint8(I*Q));

% Show images
figure;
subplot(1,3,1); imagesc(I); axis off; axis image; colormap gray; title('Original');
subplot(1,3,2); imagesc(u0); axis off; axis image; colormap gray; title(sprintf('Noisy\n PSNR:%.2f/SSIM:%.2f', noisy_psnr, noisy_ssim));
subplot(1,3,3); imagesc(u_denoised); axis off; axis image; colormap gray; title(sprintf('Poisson L1/L2 TV\n PSNR:%.2f/SSIM:%.2f', denoised_psnr, denoised_ssim));




   

