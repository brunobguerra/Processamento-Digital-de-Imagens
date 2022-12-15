%% Processamento Digital de Imagens 2021/1
% Aluno: Bruno Baptista Guerra

%% Questão 1 =================================================================================================

img_questao1 = imread('oranges.jpg');
I = double(img_questao1)/255;

R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

%Hue
numi = 1/2*((R-G)+(R-B));
denom = ((R-G).^2+((R-B).*(G-B))).^0.5;

H = acosd(numi./(denom+0.000001));
H(B>G) = 360-H(B>G);
H = H/360;

%Saturação
S=1- (3./(sum(I,3)+0.000001)).*min(I,[],3);
%Intensidade
I=sum(I,3)./3;

%HSI
HSI = zeros(size(img_questao1));
HSI(:,:,1) = H;
HSI(:,:,2) = S;
HSI(:,:,3) = I;

%Hue, Saturação and Intensidade  
H1 = HSI(:,:,1);  
S1 = HSI(:,:,2);  
I1 = HSI(:,:,3);  

H1 = H1*360;                                               
   
R1 = zeros(size(H1));  
G1 = zeros(size(H1));  
B1 = zeros(size(H1));  
RGB1 = zeros([size(H1),3]);  

%RG(0<=H<120)  
R1(H1<120) = 0.15;
G1(H1<120) = 3.*I1(H1<120)-(R1(H1<120)+B1(H1<120));
B1(H1<120) = I1(H1<120).*(1-S1(H1<120));  

%GB Sector(120<=H<240)       
H2=H1-120;  
    
R1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1-S1(H1>=120&H1<240));  
G1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1+((S1(H1>=120&H1<240).*cosd(H2(H1>=120&H1<240)))./cosd(60-H2(H1>=120&H1<240))));  
B1(H1>=120&H1<240)=3.*I1(H1>=120&H1<240)-(R1(H1>=120&H1<240)+G1(H1>=120&H1<240));      

%BR Sector(240<=H<=360)     
H2=H1-240;  

R1(H1>=240&H1<=360)=3.*I1(H1>=240&H1<=360)-(G1(H1>=240&H1<=360)+B1(H1>=240&H1<=360)); 
G1(H1>=240&H1<=360)= I1(H1>=240&H1<=360).*(1-S1(H1>=240&H1<=360));  
B1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1+((S1(H1>=240&H1<=360).*cosd(H2(H1>=240&H1<=360)))./cosd(60-H2(H1>=240&H1<=360))));  
 

%RGB Imagem  
RGB1(:,:,1) = R1;  
RGB1(:,:,2) = G1;  
RGB1(:,:,3) = B1;    
RGB1 = im2uint8(RGB1);
 
figure (1),
subplot(1,3,1), imshow(img_questao1);title('Imagem RGB');
subplot(1,3,2), imshow(HSI);title('Imagem HSI');
subplot(1,3,3), imshow(RGB1);title('Nova Imagem RGB');

%% Questão 2 =================================================================================================

img_questao2 = imread('baixo_contraste.jpg');
figure(2);
subplot(2,2,1);
imshow(img_questao2);
title('Imagem com baixo contraste');
subplot(2,2,2);
imhist(img_questao2);
title('Histograma da Imagem original');
axis tight;
h = zeros(1,256);
[r c] = size(img_questao2);
totla_no_of_pixels = r*c;
n = 0:255; 

%%
%Calculando Histograma 
for i = 1:r
    for j = 1:c
        h(img_questao2(i,j)+1) = h(img_questao2(i,j)+1)+1;
    end
end

for i = 1:256
    h(i) = h(i)/totla_no_of_pixels;
end
temp=h(1);
for i = 2:256
    temp = temp+h(i);
    h(i) = temp;
end
for i = 1:r
    for j = 1:c
        img_questao2(i,j) = round(h(img_questao2(i,j)+1)*255);
    end
end
subplot(2,2,3);
imshow(img_questao2);
title('Imagem com Histograma Equalizado');
subplot(2,2,4);
imhist(img_questao2);
axis tight;
title('Histograma Equalizatido');

%% Questão 3 =================================================================================================

img_questao3 = imread('Fig10.10(a).jpg');
n = double(img_questao3);

for i=1:size(n,1)-2
    for j=1:size(n,2)-2
        %Mascára de soble na direção x:
        Gx=((2*n(i+2,j+1)+n(i+2,j)+n(i+2,j+2))-(2*n(i,j+1)+n(i,j)+n(i,j+2)));
        %Mascára de soble na direção y:
        Gy=((2*n(i+1,j+2)+n(i,j+2)+n(i+2,j+2))-(2*n(i+1,j)+n(i,j)+n(i+2,j)));
        
        B(i,j) = sqrt(Gx.^2); %Gradiente na direção x
        
        C(i,j) = sqrt(Gy.^2); %Gradiente na direção y
        
        D(i,j) = sqrt(Gx.^2+Gy.^2);
     
    end
end

figure (3),
subplot(2,2,1);imshow(img_questao3); title('Original');
subplot(2,2,2);imshow(uint8(B)); title('Gradiente na direção x |g_x|');
subplot(2,2,3); imshow(uint8(C)); title('Gradiente na direção y |g_y|');
subplot(2,2,4); imshow(uint8(D)); title('Gradiente |g_x|+ |g_y|');

%% Questão 4  =================================================================================================

img_questao4 = imread('rice.jpg');
T = 131;
It = im2bw(img_questao4,T/255);

n = imhist(img_questao4); 
N = sum(n); 
max = 0;

for i = 1:256
    P(i) = n(i)/N; 
end

for T = 2:255     
    w0 = sum(P(1:T)); 
    w1 = sum(P(T+1:256)); 
    u0 = dot([0:T-1],P(1:T))/w0; 
    u1 = dot([T:255],P(T+1:256))/w1; 
    sigma = w0*w1*((u1-u0)^2);
    if sigma>max  
        max = sigma;
        threshold = T-1; % 
    end
end

bw = im2bw(img_questao4,threshold/255);

figure (4),
subplot(2,3,1);imshow(img_questao4); title('Imagem Original');
subplot(2,3,2);imshow(It); title('Imagem com limiarização global simples');
subplot(2,3,3);imshow(bw); title('Imagem limiarizada com Otsu');

%%
img_questao4 = double(img_questao4);
img_questao4 = img_questao4(:,:,1);
[M, N] = size(img_questao4);
img_questao4_ilu = img_questao4;
%% Iluminar imagem
for m = .8*M:M
        img_questao4_ilu(m,:) = img_questao4(m,:) + 30;
end
img_questao4_ilu = uint8(img_questao4_ilu);

T = 135;
It_ilu = im2bw(img_questao4_ilu,T/255);

n = imhist(img_questao4_ilu); 
N = sum(n); 
max = 0; 

for i = 1:256
    P(i) = n(i)/N; 
end

for T=2:255     
    w0 = sum(P(1:T));
    w1 = sum(P(T+1:256));
    u0 = dot([0:T-1],P(1:T))/w0; 
    u1 = dot([T:255],P(T+1:256))/w1; 
    sigma = w0*w1*((u1-u0)^2); 
    if sigma>max
        max = sigma; 
        threshold = T-1; 
    end
end

bw_ilu = im2bw(img_questao4_ilu,threshold/255); 

subplot(2,3,4);imshow(img_questao4_ilu); title('Imagem multiplicada por uma máscara de iluminação.');
subplot(2,3,5);imshow(It_ilu); title('Imagem iluminada com limiarização global simples');
subplot(2,3,6);imshow(bw_ilu); title('Imagem  iluminada limiarizada com Otsu');

%% Questão 5  =================================================================================================

input_image1 = double(imread('Fig8.02(a).jpg'));
input_image2 = double(imread('Fig8.02(b).jpg'));

F1 = fft2(input_image1);
magnitude1 = abs(log(1 + fftshift(F1)));


F2 = fft2(input_image2);
magnitude2 = abs(log(1 + fftshift(F2)));

psd1 = log10(abs(fftshift(fft2(input_image1))).^2 );
psd2 = log10(abs(fftshift(fft2(input_image2))).^2 );

figure(5)
subplot(2,4,1); imshow(uint8(input_image1)); title('Fósforo ordenado');
subplot(2,4,2); imshow(magnitude1,[]); title('Espectro');
subplot(2,4,3); plot(abs(fft(magnitude1(279,:)))); title('S(r)');axis tight;
subplot(2,4,4); plot(psd1(279,:)); title('S(\theta)'); axis tight;
subplot(2,4,5); imshow(uint8(input_image2)); title('Fósforo aleatório');
subplot(2,4,6); imshow(magnitude2,[]); title('Espectro');
subplot(2,4,7); plot(abs(fft(magnitude2(279,:)))); title('S(r)');axis tight;
subplot(2,4,8); plot(psd2(279,:)); title('S(\theta)');axis tight;