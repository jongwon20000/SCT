function [positions, time] = sct4(video_path, img_files, pos, target_sz, show_visualization)

%% Parameter setting
% Feature & Kernel parameters setting
interp_factor = 0.02;
kernel.sigma = 0.5;

kernel.poly_a = 1;
kernel.poly_b = 9;

features.gray = false;
features.hog = true;
features.hog_orientations = 9;
cell_size = 4;

% KCF parameters setting
padding = 1.5;  %extra area surrounding the target
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

% Attention map parameters setting
Nfo = 10;
boundary_ratio = 1/3;
salWeight = [0.3 0.3];
bSal = [1 1];

% multiple module trackers type initialization
filterPool(1).kernelType = 'gaussian';
filterPool(1).featureType = 'color';
filterPool(2).kernelType = 'polynomial';
filterPool(2).featureType = 'color';
filterPool(3).kernelType = 'gaussian';
filterPool(3).featureType = 'hog';
filterPool(4).kernelType = 'polynomial';
filterPool(4).featureType = 'hog';
% etc.
time = 0;  %to calculate FPS
positions = zeros(numel(img_files), 4);  %to calculate precision

%% Tracker initialization
%if the target is large, lower the resolution
resize_image = (sqrt(prod(target_sz)) >= 100);
if resize_image,
    pos = floor(pos / 2);
    target_sz = floor(target_sz / 2);
end

window_sz = floor(target_sz * (1 + padding));

% Initialize the constant values & maps
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
cos_window = (hann(size(yf,1)) * hann(size(yf,2))');
mask = ones(size(yf,1), size(yf,2)); % Initial mask for strong saliency map
depthBoundaryX = max(round(size(yf,2)*boundary_ratio), 3);
depthBoundaryY = max(round(size(yf,1)*boundary_ratio), 3);
mask( depthBoundaryY:(end-depthBoundaryY+1), depthBoundaryX:(end-depthBoundaryX+1) ) = 0;

% visualization initialize
if show_visualization,  %create video interface
    update_visualization = show_video(img_files, video_path, resize_image);
end



%note: variables ending with 'f' are in the Fourier domain.

%% Tracking Start
for frame = 1:numel(img_files),
    
    %load image
    im = imread([video_path img_files{frame}]); % gray image for HOG feature
    im2 = im; % color or gray image
    if size(im,3) > 1,
        im = rgb2gray(im);
    end
    if resize_image,
        im = imresize(im, 0.5);
        im2 = imresize(im2, 0.5);
    end
    
    tic()
    
    if frame > 1,
        % HOG feature extraction
        patch = get_subwindow(im, pos, window_sz);
        z = get_features(patch, features, cell_size, []);
        
        % Color/Gray intensity feature extraction
        patch2 = get_subwindow(im2, pos, window_sz);
        feature = double(imresize(patch2, [size(z,1), size(z,2)]))/255;
        if(size(feature,3) > 1) % for color image, concatenate 'LAB' space
            feature = cat(3, feature, rgb2lab(feature) / 255 + 0.5);
        end
        z2 = feature;
        
        % Attention map from color/gray feature
        if(bSal(1)==1)
            stS{1,1} = evaluate_stSaliency(z2, rf{1,1});
            stS{1,1} = (1-salWeight(1))*cos_window + salWeight(1)*stS{1,1};
        else
            stS{1,1} = cos_window;
        end
        
        % Attention map from HOG feature
        if(bSal(2)==1)
            stS{1,2} = evaluate_stSaliency(z, rf{1,2});
            stS{1,2} = (1-salWeight(2))*cos_window + salWeight(2)*stS{1,2};
        else
            stS{1,2} = cos_window;
        end
                    
        % Attention map multiplication
        zs = bsxfun(@times, z, stS{1,2});
        zs2 = bsxfun(@times, z2, stS{1,1});
        zf = fft2(zs);
        zf2 = fft2(zs2);
                        
        % response map zero setting
        response = zeros(size(yf));
        
        % Module-wise correlation filter response estimation
        for ii = 1:4
            switch multiFilters(ii).kernelType
                case 'gaussian',
                    if strcmp(multiFilters(ii).featureType, 'hog')
                        kzf = gaussian_correlation(zf, multiFilters(ii).model_xf, kernel.sigma);
                    else
                        kzf = gaussian_correlation(zf2, multiFilters(ii).model_xf, kernel.sigma);
                    end
                case 'polynomial',
                    if strcmp(multiFilters(ii).featureType, 'hog')
                        kzf = polynomial_correlation(zf, multiFilters(ii).model_xf, kernel.poly_a, kernel.poly_b);
                    else
                        kzf = polynomial_correlation(zf2, multiFilters(ii).model_xf, kernel.poly_a, kernel.poly_b);
                    end
                case 'linear',
                    if strcmp(multiFilters(ii).featureType, 'hog')
                        kzf = linear_correlation(zf, multiFilters(ii).model_xf);
                    else
                        kzf = linear_correlation(zf2, multiFilters(ii).model_xf);
                    end
            end
            aa = multiFilters(ii).model_alphaf .* kzf;
            response = response + multiFilters(ii).weight*(1/ii.^1.5)*aa;
        end
        response = real(ifft2(response));
        
        % determine the target location
        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
        if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - size(zf,1);
        end
        if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
            horiz_delta = horiz_delta - size(zf,2);
        end
        
        pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];

    end
    
    % HOG feature extraction
    patch = get_subwindow(im, pos, window_sz);
    x = get_features(patch, features, cell_size, []);
    % Color/gray intensity feature extraction
    patch2 = get_subwindow(im2, pos, window_sz);
    feature = double(imresize(patch2, [size(x,1), size(x,2)]))/255;
    if(size(feature,3) > 1)
        feature =  cat(3, feature, rgb2lab(feature) / 255 + 0.5);
    end
    x2 = feature;
        
    % attentional map estimator initialization & update
    if(frame==1)
        [rf{1,1}, stS{1,1}] = init_stSaliency(x2, mask);
        saliencyMap{1,1} = cos_window;
        [rf{1,2}, stS{1,2}] = init_stSaliency(x, mask);
        saliencyMap{1,2} = cos_window;
    else
        if(frame==2)
            bSal = [1, 1];
        end
        if(bSal(1)==1)
            [rf{1,1}, stS{1,1}] = update_stSaliency(x2, mask, rf{1,1});
        end
        if(bSal(2)==1)
            [rf{1,2}, stS{1,2}] = update_stSaliency(x, mask, rf{1,2});
        end

        if(bSal(1)==1)
            salWeight(1) = exp( -3* mean(mean( (~mask - stS{1,1}).^2 )) );
        end        
        if(bSal(2)==1)
            salWeight(2) =  exp( -3* mean(mean( (~mask - stS{1,2}).^2 )) );
        end
        
        stS2{1,1} = (1-salWeight(1))*cos_window + salWeight(1)*stS{1,1};
        stS2{1,2} = (1-salWeight(2))*cos_window + salWeight(2)*stS{1,2};
        
        % initial attention map weight
        if(frame < Nfo)
            salWeight = [0.5 0.5];
        end
        
        % exception case for attention map weight
        if( frame==Nfo && mean(mean( (1-stS2{1,1}(mask==0)).^2 )) > 0.35 ...
                && mean(mean( (1-stS2{1,2}(mask==0)).^2 )) > 0.35)
            bSal(1) = -1;
            bSal(2) = -1;
            salWeight(1) = 0;
            salWeight(2) = 0;
        end
        
        % attention map weight for full occlusion case
        if( frame>Nfo && mean(mean( (1-stS2{1,1}(mask==0)).^2 )) > 0.4)
            nn = 1;
            bSal(1) = 0;
        end        
        if( frame>Nfo && mean(mean( (1-stS2{1,2}(mask==0)).^2 )) > 0.4)
            nn2 = 1;
            bSal(2) = 0;
        end
        
        if(bSal(1)==0)
            nn = nn + 1;
            if(nn > Nfo) 
                bSal(1) = 1;
            end
        end        
        if(bSal(2)==0)
            nn2 = nn2 + 1;
            if(nn2 > Nfo) 
                bSal(2) = 1;
            end
        end
        
        % update the attention map & its estimator
        stS{1,1} = (1-salWeight(1))*cos_window + salWeight(1)*stS{1,1};        
        stS{1,2} = (1-salWeight(2))*cos_window + salWeight(2)*stS{1,2};

        saliencyMap{1,1} = saliencyMap{1,1}*(1 - interp_factor) + stS{1,1}*interp_factor;
        saliencyMap{1,2} = saliencyMap{1,2}*(1 - interp_factor) + stS{1,2}*interp_factor;
        
    end
    
    stS{1,1} = saliencyMap{1,1};
    stS{1,2} = saliencyMap{1,2};
    
    % attention map multiplication for tracking
    x = bsxfun(@times, stS{1,2}, x);
    x2 = bsxfun(@times, stS{1,1}, x2);
    xf = fft2(x);
    xf2 = fft2(x2);


    % Module-wise training
    for ii = 1:4
        switch filterPool(ii).kernelType
            case 'gaussian',
                if strcmp(filterPool(ii).featureType, 'hog')
                    filterPool(ii).kf = gaussian_correlation(xf, xf, kernel.sigma);
                else
                    filterPool(ii).kf = gaussian_correlation(xf2, xf2, kernel.sigma);
                end
            case 'polynomial',
                if strcmp(filterPool(ii).featureType, 'hog')
                    filterPool(ii).kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
                else
                    filterPool(ii).kf = polynomial_correlation(xf2, xf2, kernel.poly_a, kernel.poly_b);
                end
            case 'linear',
                if strcmp(filterPool(ii).featureType, 'hog')
                    filterPool(ii).kf = linear_correlation(xf, xf);
                else
                    filterPool(ii).kf = linear_correlation(xf2, xf2);
                end
        end
        filterPool(ii).dalphaf = 1 ./ (filterPool(ii).kf + lambda);   %equation for fast training
    end

    % Temporal association (filter update)
    if frame == 1,  %first frame, train with a single image
        for ii = 1:4
            filterPool(ii).model_dalphaf = filterPool(ii).dalphaf;
            if strcmp(filterPool(ii).featureType, 'hog')
                filterPool(ii).model_xf = xf;
            else
                filterPool(ii).model_xf = xf2;
            end
        end
    else %subsequent frames, interpolate model
        for ii = 1:4
            filterPool(ii).model_dalphaf = ...
                (1 - interp_factor) * filterPool(ii).model_dalphaf + interp_factor * filterPool(ii).dalphaf;
            if strcmp(filterPool(ii).featureType, 'hog')
                filterPool(ii).model_xf = ...
                    (1 - interp_factor) * filterPool(ii).model_xf + interp_factor * xf;
            else
                filterPool(ii).model_xf = ...
                    (1 - interp_factor) * filterPool(ii).model_xf + interp_factor * xf2;
            end
        end
    end

    
    % Estimate the priority & reliability for each module
    errs = zeros(4,4);
    errWeight = ones(size(yf));
    bin = ones(1,4);

    errMaps = zeros(size(yf,1), size(yf,2) ,4);
    for ii = 1:4
        errMaps(:,:,ii) = (real(ifft2(filterPool(ii).kf .* yf .* filterPool(ii).model_dalphaf - yf))).^2;
    end

    for jj = 1:4

        if(jj < 4)
            % estimate the module-wise error
            for ii = 1:4
                errs(jj,ii) = sqrt(sum( vec(errWeight.*errMaps(:,:,ii)) ));
            end

            % Find the next best module
            idx = find(errs(jj,:) == min(errs(jj,bin==1)));
            idx = idx(1);

            bin(idx) = 0;

            % For reliability, the error weight is estimated
            errWeight = errWeight .* exp((errMaps(:,:,idx) / max(vec(errMaps(:,:,idx)))));
            errWeight = errWeight / max(vec(errWeight));

        else
            idx = find(bin==1);
        end

        % For priority, the order of the modules changes
        multiFilters(jj).kernelType = filterPool(idx).kernelType;
        multiFilters(jj).featureType = filterPool(idx).featureType;
        multiFilters(jj).model_alphaf = yf .* filterPool(idx).model_dalphaf;
        multiFilters(jj).model_xf = filterPool(idx).model_xf;
        multiFilters(jj).weight = exp(-0.01*errs(1,idx));

    end
        
    
    %save position and timing
    positions(frame,:) = [pos([2,1]) - target_sz([2,1])/2, pos([2,1]) + target_sz([2,1])/2];
    time = time + toc();

    
    %visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        
        drawnow
        
    end
    
end


if resize_image,
    positions = positions * 2;
end
end