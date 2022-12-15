%% Processamento Digital de Imagens 2021/1
% Aluno: Bruno Baptista Guerra

clc; close all; clear all;
%% Questão 1

img_questao1 = imread('Fig10.15(a).jpg');
img_size1 = size(img_questao1);
image = img_questao1(:,:,1);
A = 110;       
B = 150;         
img_size1 = size(img_questao1);
img_final1 = zeros(img_size1(1),img_size1(2));     

for I = 1:img_size1(1)
    for J = 1:img_size1(2)
        if (A < image(I,J) && image(I,J) < B)  
            img_final1(I,J) = 0;
        else
            img_final1(I,J) = image(I,J);
        end
    end
end

figure(1)
subplot(1,2,1); imshow(uint8(img_questao1)); title('Imagem Original');
subplot(1,2,2); imshow(uint8(img_final1)); title('Imagem com o fatiamento de níveis de intensidade')

%% Questão 2
 
input_image = imread('lena.tif');
[M, N] = size(input_image);
FT_img = fft2(double(input_image));
  
% a) 
D0 = 30;
u = 0:(M-1);
idx = find(u>M/2);
u(idx) = u(idx)-M;
v = 0:(N-1);
idy = find(v>N/2);
v(idy) = v(idy)-N;  
[V, U] = meshgrid(v, u);
D = sqrt(U.^2+V.^2);
H = double(D <= D0);
G = H.*FT_img;
  
output_image_lowpass = real(ifft2(double(G)));

% b)
input_image = double(imread('lena.tif'));
filter_mask = [0 -1 0;-1 5 -1; 0 -1 0];
%% 

for I = 2:M-1
    for J = 2:N-1
        sum = 0;
        row = 0;
        col = 0;
        for k = I-1:I+1
            row = row + 1;
            col = 1;
            for l = J-1:J+1
                sum = sum + input_image(k,l)*filter_mask(row,col);               
                col = col + 1;
            end
        end
      output_image_laplacian(I,J) = sum;      
    end
end

% c)
input_image = double(imread('lena.tif'));
resp = imfilter(uint8(input_image), filter_mask, 'conv'); 

minR = min(resp(:));
maxR = max(resp(:));
resp = (resp - minR) / (maxR - minR);
sharpened = uint8(input_image) + resp;
minA = min(sharpened(:));
maxA = max(sharpened(:));
sharpened = (sharpened - minA) / (maxA - minA);
output_image_sharpened = imadjust(uint8(input_image), [0.3 0.7], []);

figure(2)
subplot(2, 2, 1), imshow(uint8(input_image)); title('Imagem Original');
subplot(2, 2, 2), imshow(uint8(output_image_lowpass), [ ]); title('Imagem com filtro passa-baixa')
subplot(2, 2, 3), imshow(uint8(output_image_laplacian), [ ]); title('Imagem com filtro laplaciano')
subplot(2, 2, 4), imshow(uint8(output_image_sharpened), [ ]); title('Imagem com máscara de nitidez')

%% Questão 3

input_image = imread('frexp_1.png');
[M, N] = size(input_image);

output_image_downsample = input_image(1:2:M, 1:2:N);

for I = 1:M
    for J = 1:N
        rmin = max(1,I-1);
        rmax = min(M,I+1);
        cmin = max(1,J-1);
        cmax = min(N,J+1);
        temp = input_image(rmin:rmax, cmin:cmax);
        output_image_smoothed(I,J) = mean(temp(:));
    end
end

figure(3)
imshow(input_image); title('Imagem Original');

figure(4)
subplot(1,2,1); imshow(uint8(output_image_downsample)); title('Imagem Reduzida')
subplot(1,2,2); imshow(uint8(output_image_smoothed)); title('Imagem Suavizada')

%% Questão 4

input_image1 = double(imread('lena.tif'));
input_image2 = double(imread('elaine.tif'));


F1 = fft2(input_image1);
magnitude1 = abs(log(1 + fftshift(F1)));
phase1 = atan2(imag(F1),real(F1));

F2 = fft2(input_image2);
magnitude2 = abs(log(1 + fftshift(F2)));
phase2 = atan2(imag(F2),real(F2));

figure(5)
subplot(2,3,1); imshow(uint8(input_image1)); title('Imagem Original');
subplot(2,3,2); imshow(magnitude1,[]); title('Espectro');
subplot(2,3,3); imshow(phase1,[]); title('Fase');
subplot(2,3,4); imshow(uint8(input_image2)); title('Imagem Original');
subplot(2,3,5); imshow(magnitude2,[]); title('Espectro');
subplot(2,3,6); imshow(phase2,[]); title('Fase');
% 
combined1 =  abs(F1) .* exp(j*phase2);
img_combined1 = real(ifft2(combined1));

combined2 =  abs(F2) .* exp(j*phase1);
img_combined2 = real(ifft2(combined2));

figure(6)
subplot(2,3,1); imshow(uint8(img_combined1)); title('Lenna Alterada');
subplot(2,3,2); imshow(magnitude1,[]); title('Espectro da Lenna');
subplot(2,3,3); imshow(phase1,[]); title('Fase da Elaine');
subplot(2,3,4); imshow(uint8(img_combined2)); title('Elaine Alterada');
subplot(2,3,5); imshow(magnitude2,[]); title('Espectro da Elaine');
subplot(2,3,6); imshow(phase2,[]); title('Fase da Lenna');


%% Questão 5

input_image = imread('lena.tif');
[M, N] = size(input_image);
FT_img = fft2(double(input_image));
  
n = 1;
D0 = 20; 

u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
  
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2); 
H = 1./(1 + (D./D0).^(2*n));
G = H.*FT_img;

output_image1 = real(ifft2(double(G)));

%%
n = 1; 
D0 = 60; 

u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
  
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);  
H = 1./(1 + (D./D0).^(2*n));
G = H.*FT_img;

output_image2 = real(ifft2(double(G)));

%%
n = 8;  
D0 = 20; 

u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
  
[V, U] = meshgrid(v, u); 
D = sqrt(U.^2 + V.^2);
H = 1./(1 + (D./D0).^(2*n));
G = H.*FT_img;

output_image3 = real(ifft2(double(G)));

%%
n = 8;  
D0 = 60; 

u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
  
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2); 
H = 1./(1 + (D./D0).^(2*n));
  
G = H.*FT_img;

output_image4 = real(ifft2(double(G)));

figure(7)
subplot(2,2,1), imshow(uint8(output_image1)); title('N = 1 e D0 = 20');
subplot(2,2,2), imshow(uint8(output_image2)); title('N = 1 e D0 = 60');
subplot(2,2,3), imshow(uint8(output_image3)); title('N = 8 e D0 = 20');
subplot(2,2,4), imshow(uint8(output_image4)); title('N = 8 e D0 = 60');

%% Questão 6

input_original = imread('original.tif');
input_image1 = imread('ruidosa1.tif');
[m1,n1] = size(input_image1);
input_image2 = imread('ruidosa2.tif');
[m2,n2] = size(input_image2);

%% Filtro de Média 3x3 - Ruidosa 1

b = 3;
z = ones(b);
[p,q] = size(z);
w = 1:p;
x = round(median(w));
anz = zeros(m1+2*(x-1),n1+2*(x-1));
for i = x:(m1+(x-1))
    for j = x:(n1+(x-1))
        anz(i,j) = input_image1(i-(x-1),j-(x-1));
    end
end
sum = 0;
x = 0;
y = 0;
for i = 1:m1
    for j = 1:n1
        for k = 1:p
            for l = 1:q 
                sum = sum + anz(i+x,j+y)*z(k,l);
                y = y + 1;
            end
            y = 0;
            x = x + 1;
        end
        x = 0;
        output_image1_mean3x3(i,j) = (1/(p*q))*(sum);
        sum = 0;
    end
end

%% Filtro de Média 11x11 - Ruidosa 1

b = 11;
z = ones(b);
[p,q] = size(z);
w = 1:p;
x = round(median(w));
anz = zeros(m1+2*(x-1),n1+2*(x-1));
for i = x:(m1+(x-1))
    for j = x:(n1+(x-1))
        anz(i,j) = input_image1(i-(x-1),j-(x-1));
    end
end
sum = 0;
x = 0;
y = 0;
for i = 1:m1
    for j = 1:n1
        for k = 1:p
            for l = 1:q 
                sum = sum + anz(i+x,j+y)*z(k,l);
                y = y + 1;
            end
            y = 0;
            x = x + 1;
        end
        x = 0;
        output_image1_mean11x11(i,j) = (1/(p*q))*(sum);
        sum = 0;
    end
end

%% Filtro de Média 3x3 - Ruidosa 2

b = 3;
z = ones(b);
[p,q] = size(z);
w = 1:p;
x = round(median(w));
anz = zeros(m2 + 2*(x-1),n2+2*(x-1));
for i = x:(m1+(x-1))
    for j = x:(n2+(x-1))
        anz(i,j) = input_image2(i-(x-1),j-(x-1));
    end
end
sum = 0;
x = 0;
y = 0;
for i = 1:m2
    for j = 1:n2
        for k = 1:p
            for l = 1:q 
                sum = sum + anz(i+x,j+y)*z(k,l);
                y = y + 1;
            end
            y = 0;
            x = x+1;
        end
        x = 0;
        output_image2_mean3x3(i,j) = (1/(p*q))*(sum);
        sum = 0;
    end
end


%% Filtro de Média 11x11 - Ruidosa 2

b = 11;
z = ones(b);
[p,q] = size(z);
w = 1:p;
x = round(median(w));
anz = zeros(m2+2*(x-1),n2+2*(x-1));
for i = x:(m2+(x-1))
    for j = x:(n2+(x-1))
        anz(i,j) = input_image2(i-(x-1),j-(x-1));
    end
end
sum = 0;
x = 0;
y = 0;
for i = 1:m2
    for j = 1:n2
        for k = 1:p
            for l = 1:q 
                sum = sum + anz(i+x,j+y)*z(k,l);
                y = y + 1;
            end
            y = 0;
            x = x+1;
        end
        x = 0;
        output_image2_mean11x11(i,j) = (1/(p*q))*(sum);
        sum = 0;
    end
end 

%% PLOTAGEM

figure(8)
subplot(2,3,1); imshow(uint8(input_image1)); title('Ruidosa 1');
subplot(2,3,2); imshow(uint8(output_image1_mean3x3)); title('Filtro de Média - N = 3');
subplot(2,3,3); imshow(uint8(output_image1_mean11x11)); title('Filtro de Média - N = 11');
subplot(2,3,4); imshow(uint8(input_image2)); title('Ruidosa 2');
subplot(2,3,5); imshow(uint8(output_image2_mean3x3));title('Filtro de Média - N = 3');
subplot(2,3,6); imshow(uint8(output_image2_mean11x11)); title('Filtro de Média - N = 11');

%% PSNR

ref = uint8(input_original);

%Ruidosa 1
A = uint8(input_image1);
squaredErrorImage = (double(input_original) - double(input_image1)) .^ 2;
[rows, columns] = size(input_image1);
MSE = sum(sum(squaredErrorImage)) / (rows * columns);
PSNR = 10 * log10( 256^2 / MSE);
% [peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R1 = %0.4f', PSNR);

A = uint8(output_image1_mean3x3);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R1M3 = %0.4f', peaksnr);

A = uint8(output_image1_mean11x11);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R1M11 = %0.4f', peaksnr);

%Ruidosa 2
A = uint8(input_image2);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R2 = %0.4f', peaksnr);

A = uint8(output_image2_mean3x3);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R2M3 = %0.4f', peaksnr);

A = uint8(output_image2_mean11x11);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R2M11 = %0.4f', peaksnr);

%% Filtro de Mediana 3x3 - Ruidosa 1

modifyA = zeros(size(input_image1)+2);
B = zeros(size(input_image1));

for x = 1:size(input_image1,1)
    for y = 1:size(input_image1,2)
        modifyA(x+1,y+1) = input_image1(x,y);
    end
end

for I = 1:size(modifyA,1)-2
    for J = 1:size(modifyA,2)-2
        window = zeros(9,1);
        inc = 1;
        for x = 1:3
            for y = 1:3
                window(inc) = modifyA(I+x-1,J+y-1);
                inc = inc + 1;
            end
        end
       
        med = sort(window);
        B(I,J) = med(5);
    end
end
output_image1_median3x3 = uint8(B);

%% Filtro de Mediana 11x11 - Ruidosa 1

M = 11;
N = 11;

modifyA = padarray(input_image1,[floor(M/2),floor(N/2)]);
B = zeros([size(input_image1,1) size(input_image1,2)]);
med_indx = round((M*N)/2);

for i = 1:size(modifyA,1)-(M-1)
    for j = 1:size(modifyA,2)-(N-1)
        temp = modifyA(i:i+(M-1),j:j+(N-1),:);
        tmp_sort = sort(temp(:));
        B(i,j) = tmp_sort(med_indx);
    end
end

output_image1_median11x11 = uint8(B);

%% Filtro de Mediana 3x3 - Ruidosa 2

modifyA = zeros(size(input_image2)+2);
B = zeros(size(input_image2));

for x = 1:size(input_image2,1)
    for y = 1:size(input_image2,2)
        modifyA(x+1,y+1) = input_image2(x,y);
    end
end

for I = 1:size(modifyA,1)-2
    for J = 1:size(modifyA,2)-2
        window = zeros(9,1);
        inc = 1;
        for x = 1:3
            for y = 1:3
                window(inc) = modifyA(I+x-1,J+y-1);
                inc = inc+1;
            end
        end
        med = sort(window);
        B(I,J) = med(5);
    end
end
output_image2_median3x3 = uint8(B);

%% Filtro de Mediana 11x11 - Ruidosa 2

M = 11;
N = 11;

modifyA = padarray(input_image2,[floor(M/2),floor(N/2)]);
B = zeros([size(input_image2,1) size(input_image2,2)]);
med_indx = round((M*N)/2); 

for i = 1:size(modifyA,1)-(M-1)
    for j = 1:size(modifyA,2)-(N-1)
        temp = modifyA(i:i+(M-1),j:j+(N-1),:);
        tmp_sort = sort(temp(:));
        B(i,j) = tmp_sort(med_indx);
    end
end

output_image2_median11x11 = uint8(B);

%% PLOTAGEM 

figure(9)
subplot(2,3,1); imshow(uint8(input_image1)); title('Ruidosa 1');
subplot(2,3,2); imshow(output_image1_median3x3); title('Filtro de Mediana - N = 3');
subplot(2,3,3); imshow(output_image1_median11x11); title('Filtro de Mediana - N = 11');
subplot(2,3,4); imshow(uint8(input_image2)); title('Ruidosa 2');
subplot(2,3,5); imshow(output_image2_median3x3);title('Filtro de Mediana - N = 3');
subplot(2,3,6); imshow(output_image2_median11x11); title('Filtro de Mediana - N = 11');


%% PSNR

ref = uint8(input_original);

%Ruidosa 1
A = uint8(input_image1);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R1 = %0.4f', peaksnr);


A = uint8(output_image1_median3x3);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R1M3 = %0.4f', peaksnr);


A = uint8(output_image1_median11x11);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R1M11 = %0.4f', peaksnr);

% Ruidosa 2
A = uint8(input_image2);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R2 = %0.4f', peaksnr);


A = uint8(output_image2_median3x3);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R2M3 = %0.4f', peaksnr);


A = uint8(output_image2_median11x11);
[peaksnr, snr] = psnr(A, ref);
fprintf('\n PSNR_R2M11 = %0.4f', peaksnr);