function results=calculate_PSNR_512(ip1,tar1,est1)
%psnr
img_size = 512;
smax = 1;
ip11 = double(ip1);
tar11 = double(tar1);
mse_op = power((ip11-tar11),2);
mse2_op= sum(mse_op(:))/(img_size*img_size);
snr_op = smax*smax/mse2_op;
psnr_op = 10*log10(snr_op);
%disp('psnr_ip');
%display(psnr_op);

est11 = double(est1);
mse_op2 = power((est11-tar11),2);
mse2_op2= sum(mse_op2(:))/(img_size*img_size);
snr_op2 = smax*smax/mse2_op2;
psnr_op2 = 10*log10(snr_op2);
%disp('psnr_est');
%display(psnr_op2);
%results=zeros(1,2);
results = [psnr_op,psnr_op2];
end