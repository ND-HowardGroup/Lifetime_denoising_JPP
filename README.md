# Lifetime_denoising_JPP


# For lifetime denoising code:

Input (noisy) HSV composite lifetime Image: In the HSV format, where hue and value are mapped to lifetime and intensity respectively.

![](Final%20results/image_input_PM_FLIM_073.png)

Denoised composte lifetime Image: From CNN ML model (with autoencoder architecture): In the HSV format, where hue and value are mapped to lifetime and intensity respectively.

![](Final%20results/estimated_128_rgb_3d.png)

Target composite lifetime Image: In the HSV format, where hue and value are mapped to lifetime and intensity respectively.

![](Final%20results/target_rgb_3d.png)


# ML model:

Input image dimensions are: 512x512 slices into 256x256 image:
G and S images are pass through the trained Noise2Noise model and denoised images are 256x256 slices joined together to form complete 512x512 denoised $G$ and $S$ images. Here to represent the lifetime images, the HSV format where hue and value are mapped to lifetime and intensity respectively is used (always we use here the intensity image is the acquired raw noisy intensity image). 



## License & Copyright
© 2019 Varun Mannam, University of Notre Dame

Licensed under the [Apache License 2.0](https://github.com/varunmannam/Lifetime_denoising_JPP/blob/master/LICENSE)
