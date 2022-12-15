function Y = avaliarMadeira(IMG)
    IMG = double(IMG);
    
    %IMG = convolution(IMG,11,'mean',1);
    %IMG = convolution(IMG,3,'sobelx',1);
    IMG = binariz(IMG,[0,49]);
    IMG = convolution(IMG,5,'median',1);
    
    Y = uint8(IMG);
end