function Y = binariz(IMG,INTERVAL)
    IMG = double(IMG);
    [M,N] = size(IMG);
    for m = 1:M
        for n = 1:N
            if ( INTERVAL(1) < IMG(m,n) ) && ( IMG(m,n) < INTERVAL(2) )
                IMG(m,n) = 255;
            else
                IMG(m,n) = 0;
            end
        end
    end
    Y = uint8(IMG);
end