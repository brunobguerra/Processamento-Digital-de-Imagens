function [x2,y2] = centroide(x1,y1,MASK)
    arguments
       x1
       y1
       MASK
    end
    MASK = double(MASK)/255;
    [~,N] = size(MASK);
    
    Cx = 0; Cy = 0; C = (N-1)/2;
    
    S = sum(sum(MASK));
    for k = 1:N
        Cx = Cx + (x1+k-1)*sum(MASK(k,:));
        Cy = Cy + (y1+k-1)*sum(MASK(:,k));
    end
    x2 = round(Cx/S) - C;
    y2 = round(Cy/S) - C;
    
end