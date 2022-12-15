function Y = convolution(IMG,N,OPTION,C)
    arguments
       IMG
       N = 3
       OPTION = 'mean'
       C = -1
    end
    % Y = convolution(IMG,N,OPTION,C)
    % IMG is the image to be convolutioned.
    % N is the mask dimension, it must be odd.
    % OPTION is filter type.
    % OPTION can be: 'mean' filter, 'laplacian' filter or 'unsharp' mask.
    % OPTION can be: 'median' filter.
    % If C = 1, the operation is correlation.
    
    % Image
    IMG = double(IMG);
    
    % Mask
    if not(rem(N,2)) % if N is not odd
        error('N must be odd')
    end
    
    switch OPTION
        case 'laplacian'
            MASK = -1*ones(N,N);
            MASK(1,1) = 0; MASK(3,1) = 0; MASK(1,3) = 0; MASK(3,3) = 0;
            MASK(2,2) = 5;
        case 'sobelx'
            MASK = [1 2 1;0 0 0;-1 -2 -1];
        otherwise
            MASK = ones(N,N);
    end
    
    % Tamanho da imagem e mascara
    MN = size(IMG);
    B = (N-1)/2;
    
    % Bordas de pixels pretas
    IMG_0 = zeros(MN+(N-1));
    IMG_0(ceil(N/2):MN(1)+B,ceil(N/2):MN(2)+B) = IMG;
    IMG_AUX = IMG;
    
    if C == -1
        MASK = MASK(N:-1:1,N:-1:1);
    end
    
    
    for x = 1:MN(1)
        for y = 1:MN(2)
            switch OPTION
                case 'median'
                    I = IMG_0(x:x+(N-1),y:y+(N-1)).*MASK;
                    IMG_AUX(x,y) = median(reshape(I,[1,N*N]));
                otherwise
                    IMG_AUX(x,y) = sum(sum(IMG_0(x:x+(N-1),y:y+(N-1)).*MASK));
            end
        end
    end
    
    switch OPTION
        case 'mean'
            IMG = IMG_AUX/(N*N);
            
        case 'median'
            IMG = IMG_AUX;
            
        case 'laplacian'
            IMG = IMG_AUX;
        
        case 'sobelx'
            IMG = IMG_AUX;
            
        case 'unsharp'
            IMG = (IMG - IMG_AUX/(N*N)) + IMG;
            
        otherwise
        	warning('Unexpected type of mask.')
            return
    end
    
    Y = uint8(IMG);
end