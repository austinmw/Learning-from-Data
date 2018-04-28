function M = colormat(p)

    M = zeros(length(p),3);
    for i=1:length(p)
        switch p(i)
            case 1
                M(i,:) = [1 0 0];
            case 2
                M(i,:) = [0 0 1];
            case 3
                M(i,:) = [0 1 0];
            case 4
                M(i,:) = [0 0 0];
            otherwise
                warning('Unexpected value!');
        end
    end   
    
end

