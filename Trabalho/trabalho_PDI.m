bom1 = imread('Bom/014.png');
bom2 = imread('Bom/036.png');
bom3 = imread('Bom/037.png');
bom4 = imread('Bom/053.png');
bom5 = imread('Bom/193.png');
ruim1 = imread('Defeito/000.png');
ruim2 = imread('Defeito/001.png');
ruim3 = imread('Defeito/002.png');
ruim4 = imread('Defeito/004.png');
ruim5 = imread('Defeito/006.png');
ruim6 = imread('Defeito/007.png');
ruim7 = imread('Defeito/009.png');
images = {bom1,bom2,bom3,bom4,bom5,ruim1,ruim2,ruim3,ruim4,ruim5,ruim6,ruim7};
return
%%
tic
k = 12;
[M,N] = size(bom1);
mad_marc = uint8(zeros(M,N));

%for k = 1:(5+7)
figure
% subplot(1,2,1)
    img = avaliarMadeira(images{k});
    squares = marcador2(img);
    mad_marc(:,:,1) = images{k} + squares;
    mad_marc(:,:,2) = images{k};
    mad_marc(:,:,3) = images{k};
    imshow(mad_marc)
    if sum(sum(img))
        title('Defeito')
    else
        title('Normal')
    end
% subplot(1,2,2)
%     imshow(img + squares);
%end
toc

    
    
    
    
    
    
    
    