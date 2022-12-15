function Y = marcador(IMG,n)
    arguments
       IMG
       n = 3
    end
    % Y = marcador(IMG,N)
    % IMG is the image to be convolutioned.
    % N is the mask dimension, it must be odd.
    
    % Image
    IMG = double(IMG);
    
    % Tamanho da imagem e mascara
    [M,N] = size(IMG);
    IMG_AUX = zeros(M,N);
    
    B = 3; % borda dos marcador em pixels (PAR)
    L = 24; % comprimento adicional do marcador
    
    for x = 1:M-(n-1)
        for y = 1:N-(n-1)
            MASK = IMG( x:x+(n-1), y:y+(n-1) );
            C = sum(sum( MASK ));
            if C ~= 0                       % encontrar um ponto ou borda
                x0 = x; y0 = y;
                C1 = 0; C2 = C;
                nn = n; 
                x1 = x0; y1 = y0;
                while C2 > C1               % aumenta a área de pontos conexos
                    C1 = C2;
                    
                    %centroide(MASK)
                    
                    x1 = max(1, x1 - L/4);  % DESLOCA MARCADOR
                    y1 = max(1, y1 - L/2);  % DESLOCA MARCADOR
                    nn = nn + L;            % AUMENTA MARCADOR
                    
                    MASK = IMG( x1:x1+(nn-1), y1:y1+(nn-1) );
                    C2 = sum(sum( MASK ));
                end
                
                % SUP:
                IMG_AUX(x1:x1+(nn-1), y1:y1+B)                  = 255;
                
                % INF:
                IMG_AUX(x1:x1+(nn-1), y1+(nn-1)-B:y1+(nn-1))    = 255;
                
                % ESQ:
                IMG_AUX(max(1,x1-B):x1, y1:y1+(nn-1))                  = 255;
                
                % DIR:
                IMG_AUX(x1+(nn-1):x1+(nn-1)+B, y1:y1+(nn-1))   	= 255;
                
                IMG(x1:x1+(nn-1), y1:y1+(nn-1)) = 0;
            end
        end
    end
    
    IMG = IMG_AUX;
    Y = uint8(IMG);
end