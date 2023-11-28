function results = tracker(params, video)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im, im_t] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Correct max number of samples
params.nSamples = min(params.nSamples, seq.num_frames);

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

if size(im_t,3) > 1 && is_color_image == false
    im_t = im_t(:,:,1);
end


% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('ECO:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);

% Get feature specific parameters
feature_params = init_feature_params(features, feature_info);
feature_extract_info = get_feature_extract_info(features);

% Set the sample feature dimension
if params.use_projection_matrix
    sample_dim = feature_params.compressed_dim;
else
    sample_dim = feature_dim;
end

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% How much each feature block has to be padded to the obtain output_sz
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);

% Compute the Fourier series indices and their transposes
ky = cellfun(@(sz) (-ceil((sz(1) - 1)/2) : floor((sz(1) - 1)/2))', filter_sz_cell, 'uniformoutput', false);
kx = cellfun(@(sz) -ceil((sz(2) - 1)/2) : 0, filter_sz_cell, 'uniformoutput', false);

% construct the Gaussian label function using Poisson formula
sig_y = sqrt(prod(floor(base_target_sz))) * params.output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = cellfun(@(ky) single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2)), ky, 'uniformoutput', false);
yf_x = cellfun(@(kx) single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2)), kx, 'uniformoutput', false);
yf = cellfun(@(yf_y, yf_x) cast(yf_y * yf_x, 'like', params.data_type), yf_y, yf_x, 'uniformoutput', false);

% construct cosine window
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
reg_filter = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

% Compute the energy of the filter (used for preconditioner)
reg_energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), reg_filter, 'uniformoutput', false);


if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
else
    % Use the translation filter to estimate the scale.
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
end

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% Set conjugate gradient uptions
init_CG_opts.CG_use_FR = true;
init_CG_opts.tol = 1e-6;
init_CG_opts.CG_standard_alpha = true;
init_CG_opts.debug = params.debug;
CG_opts.CG_use_FR = params.CG_use_FR;
CG_opts.tol = 1e-6;
CG_opts.CG_standard_alpha = params.CG_standard_alpha;
CG_opts.debug = params.debug;
if params.CG_forgetting_rate == Inf || params.learning_rate >= 1
    CG_opts.init_forget_factor = 0;
else
    CG_opts.init_forget_factor = (1-params.learning_rate)^params.CG_forgetting_rate;
end

seq.time = 0;

% Initialize and allocate
prior_weights = zeros(params.nSamples,1, 'single');
prior_weights_t = prior_weights;

sample_weights = cast(prior_weights, 'like', params.data_type);
sample_weights_t = sample_weights;

samplesf = cell(1, 1, num_feature_blocks);
samplesf_t = samplesf;
if params.use_gpu
    % In the GPU version, the data is stored in a more normal way since we
    % dont have to use mtimesx.
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,sample_dim(k),params.nSamples, 'like', params.data_type_complex);
    end
else
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
        samplesf_t{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
    end
end

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);
scores_fs_feat_t = scores_fs_feat;

% Distance matrix stores the square of the euclidean distance between each pair of
% samples. Initialise it to inf
distance_matrix = inf(params.nSamples, 'single');
distance_matrix_t = distance_matrix;

% Kernel matrix, used to update distance matrix
gram_matrix = inf(params.nSamples, 'single');
gram_matrix_t = gram_matrix;

latest_ind = [];
latest_ind_t = [];
frames_since_last_train = inf;
num_training_samples = 0;

% Find the minimum allowed sample weight. Samples are discarded if their weights become lower
params.minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*params.nSamples);

res_norms = [];
res_norms_t = res_norms;
residuals_pcg = [];

while true
    % Read image
    if seq.frame > 0
        [seq, im, im_t] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
        if size(im_t,3) > 1 && is_color_image == false
            im_t = im_t(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            det_sample_pos = sample_pos;
            sample_scale = currentScaleFactor*scaleFactors;
            
            %im_t = im;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            xt_t = extract_features(im_t, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            
            % Project sample
            xt_proj = project_sample(xt, projection_matrix);
            xt_proj_t = project_sample(xt_t, projection_matrix_t);
            
            % Do windowing of features
            xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
            xt_proj_t = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj_t, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
            xtf_proj_t = cellfun(@cfft2, xt_proj_t, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
            xtf_proj_t = interpolate_dft(xtf_proj_t, interp1_fs, interp2_fs);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
            scores_fs_feat_t{k1} = sum(bsxfun(@times, hf_full_t{k1}, xtf_proj_t{k1}), 3);
            
            scores_fs_sum = scores_fs_feat{k1};
            scores_fs_sum_t = scores_fs_feat_t{k1};
            for k = block_inds
                scores_fs_feat{k} = sum(bsxfun(@times, hf_full{k}, xtf_proj{k}), 3);
                scores_fs_feat_t{k} = sum(bsxfun(@times, hf_full_t{k}, xtf_proj_t{k}), 3);
                
                scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
                    scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
                    scores_fs_feat{k};
                scores_fs_sum_t(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
                    scores_fs_sum_t(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
                    scores_fs_feat_t{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            scores_fs_t = permute(gather(scores_fs_sum_t), [1 2 4 3]);
            
            
            [~, ~, scale_ind] = optimize_scores(scores_fs, params.newton_iterations);
            [~, ~, scale_ind_t] = optimize_scores(scores_fs_t, params.newton_iterations);
            sampled_scores_display_rgb = 10.*fftshift(sample_fs(scores_fs(:,:,scale_ind), 1*output_sz));
            sampled_scores_display_t = 10.*fftshift(sample_fs(scores_fs_t(:,:,scale_ind_t), 1*output_sz));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  PSR_RGB   %%%%%%%%%%%%%%%%%%%%%%%%%%%
            mres_rgb = max(max(sampled_scores_display_rgb));
            u_rgb = mean2(sampled_scores_display_rgb);
            f_rgb = std2(sampled_scores_display_rgb);
            PSR_rgb = (mres_rgb -u_rgb)/f_rgb;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  PSR_T   %%%%%%%%%%%%%%%%%%%%%%%%%%%
            mres_t = max(max(sampled_scores_display_t));
            u_t = mean2(sampled_scores_display_t);
            f_t = std2(sampled_scores_display_t);
            PSR_t = (mres_t -u_t)/f_t;


            %%%%%%%%%%%%%%%% 0<w_rgb, 0<w_t, w_rgb+w_t=1 £¬ Initial Solution %%%%%%%%%%%%%%%%%%
            for w_rgb = 0.0:0.1:1
                mres = max(max(w_rgb.*sampled_scores_display_rgb+(1-w_rgb).*sampled_scores_display_t));
                u = mean2(w_rgb.*sampled_scores_display_rgb+(1-w_rgb).*sampled_scores_display_t);
                f = std2(w_rgb.*sampled_scores_display_rgb+(1-w_rgb).*sampled_scores_display_t);
                i = round(10*w_rgb+1);
                j = round(10*(1-w_rgb)+1);
                PSR(i,j) = (mres-u)/f;
            end

            [times_x, times_y] = find (PSR == max(max(PSR)));
            W_rgb((seq.frame+1)/2) = (11-min(times_y))*0.1;
            W_t((seq.frame+1)/2) = (11-max(times_x))*0.1;
            weight_rgb = W_rgb((seq.frame+1)/2);
            weight_t = W_t((seq.frame+1)/2); 
            u_r = mean2(sampled_scores_display_rgb);
            u_t = mean2(sampled_scores_display_t);
            N=(size(sampled_scores_display_t,1))^2;
         
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Optimization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            [x,fval]= fmincon(@(x)objective(u_r,u_t,N,sampled_scores_display_rgb,sampled_scores_display_t,x),[weight_rgb,weight_t],[],[],[1 1],1,[0,0],[],[]);
            weight_rgb_non = x(1);
            weight_t_non = x(2);

            scores_fs = weight_rgb_non.*scores_fs + weight_t_non.*scores_fs_t;
            
            % Optimize the continuous score function with Newton's method.
            [trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);
            scale_change_factor = scaleFactors(scale_ind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Do scale tracking with the scale filter
            if nScales > 0 && params.use_scale_filter
                scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
            end
            
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Extract sample and init projection matrix
    if seq.frame == 1
        % Extract image region for training sample
        sample_pos = round(pos);
        sample_scale = currentScaleFactor;
        
        %im_t = im;
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        xl_t = extract_features(im_t, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        % Do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        xlw_t = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_t, cos_window, 'uniformoutput', false);
        % Compute the fourier series
        xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
        xlf_t = cellfun(@cfft2, xlw_t, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
        xlf_t = interpolate_dft(xlf_t, interp1_fs, interp2_fs);
        
        % New sample to be added
        xlf = compact_fourier_coeff(xlf);
        xlf_t = compact_fourier_coeff(xlf_t);
        
        % Shift sample
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        
        xlf = shift_sample(xlf, shift_samp, kx, ky);
        xlf_t = shift_sample(xlf_t, shift_samp, kx, ky);
        
        % Init the projection matrix
        projection_matrix = init_projection_matrix(xl, sample_dim, params);
        projection_matrix_t = init_projection_matrix(xl_t, sample_dim, params);
        
        % Project sample
        xlf_proj = project_sample(xlf, projection_matrix);
        xlf_proj_t = project_sample(xlf_t, projection_matrix_t);
        
        clear xlw
    elseif params.learning_rate > 0
        if ~params.use_detection_sample
            % Extract image region for training sample
            sample_pos = round(pos);
            sample_scale = currentScaleFactor;
            xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            xl_t = extract_features(im_t, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % Project sample
            xl_proj = project_sample(xl, projection_matrix);
            xl_proj_t = project_sample(xl_t, projection_matrix);
            
            % Do windowing of features
            xl_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
            xl_proj_t = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj_t, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xlf1_proj = cellfun(@cfft2, xl_proj, 'uniformoutput', false);
            xlf1_proj_t = cellfun(@cfft2, xl_proj_t, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xlf1_proj = interpolate_dft(xlf1_proj, interp1_fs, interp2_fs);
            xlf1_proj_t = interpolate_dft(xlf1_proj_t, interp1_fs, interp2_fs);
            
            % New sample to be added
            xlf_proj = compact_fourier_coeff(xlf1_proj);
            xlf_proj_t = compact_fourier_coeff(xlf1_proj_t);
        else
            if params.debug
                % Only for visualization
                xl = cellfun(@(xt) xt(:,:,:,scale_ind), xt, 'uniformoutput', false);
            end
            
            % Use the sample that was used for detection
            sample_scale = sample_scale(scale_ind);
            xlf_proj = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,scale_ind), xtf_proj, 'uniformoutput', false);
            xlf_proj_t = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,scale_ind), xtf_proj_t, 'uniformoutput', false);
        end
        
        % Shift the sample so that the target is centered
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);
        xlf_proj_t = shift_sample(xlf_proj_t, shift_samp, kx, ky);
    end
    
    % The permuted sample is only needed for the CPU implementation
    if ~params.use_gpu
        xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
        xlf_proj_perm_t = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj_t, 'uniformoutput', false);
    end
    
    if params.use_sample_merge
        % Update the samplesf to include the new sample. The distance
        % matrix, kernel matrix and prior weight are also updated
        if params.use_gpu
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model_gpu(samplesf, xlf_proj, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        else
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model(samplesf, xlf_proj_perm, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
            
            [merged_sample_t, new_sample_t, merged_sample_id_t, new_sample_id_t, distance_matrix_t, gram_matrix_t, prior_weights_t] = ...
                update_sample_space_model(samplesf, xlf_proj_perm_t, distance_matrix_t, gram_matrix_t, prior_weights_t,...
                num_training_samples,params);
        end
        
        if num_training_samples < params.nSamples
            num_training_samples = num_training_samples + 1;
        end
    else
        % Do the traditional adding of a training sample and weight update
        % of C-COT
        [prior_weights, replace_ind] = update_prior_weights(prior_weights, gather(sample_weights), latest_ind, seq.frame, params);
        [prior_weights_t, replace_ind_t] = update_prior_weights(prior_weights_t, gather(sample_weights_t), latest_ind_t, seq.frame, params);
        
        latest_ind = replace_ind;
        latest_ind_t = replace_ind_t;
        
        merged_sample_id = 0;
        merged_sample_id_t = 0;
        new_sample_id = replace_ind;
        new_sample_id_t = replace_ind_t;
        if params.use_gpu
            new_sample = xlf_proj;
        else
            new_sample = xlf_proj_perm;
        end
    end
    
    if seq.frame > 1 && params.learning_rate > 0 || seq.frame == 1 && ~params.update_projection_matrix
        % Insert the new training sample
        for k = 1:num_feature_blocks
            if params.use_gpu
                if merged_sample_id > 0
                    samplesf{k}(:,:,:,merged_sample_id) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(:,:,:,new_sample_id) = new_sample{k};
                end
            else
                if merged_sample_id > 0
                    samplesf{k}(merged_sample_id,:,:,:) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(new_sample_id,:,:,:) = new_sample{k};
                end
            end
        end
        for k = 1:num_feature_blocks
            if params.use_gpu
                if merged_sample_id_t > 0
                    samplesf_t{k}(:,:,:,merged_sample_id_t) = merged_sample_t{k};
                end
                if new_sample_id_t > 0
                    samplesf_t{k}(:,:,:,new_sample_id_t) = new_sample_t{k};
                end
            else
                if merged_sample_id_t > 0
                    samplesf_t{k}(merged_sample_id_t,:,:,:) = merged_sample_t{k};
                end
                if new_sample_id_t > 0
                    samplesf_t{k}(new_sample_id_t,:,:,:) = new_sample_t{k};
                end
            end
        end
    end
    
    sample_weights = cast(prior_weights, 'like', params.data_type);
    sample_weights_t = cast(prior_weights_t, 'like', params.data_type);
    
    train_tracker = (seq.frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
    
    if train_tracker 
        % Used for preconditioning
        new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
        new_sample_energy_t = cellfun(@(xlf_t) abs(xlf_t .* conj(xlf_t)), xlf_proj_t, 'uniformoutput', false);
        
        if seq.frame == 1
            % Initialize stuff for the filter learning
            
            % Initialize Conjugate Gradient parameters
            sample_energy = new_sample_energy;
            sample_energy_t = new_sample_energy_t;
            
            CG_state = [];
            CG_state_t = [];
            
            if params.update_projection_matrix
                % Number of CG iterations per GN iteration
                init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
                
                hf = cell(2,1,num_feature_blocks);
                hf_t = hf;
                
                proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix, yf, 'uniformoutput', false);
                proj_energy_t = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix_t, yf, 'uniformoutput', false);
            else
                CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
                
                hf = cell(1,1,num_feature_blocks);
                hf_t = hf;
            end
            
            % Initialize the filter with zeros
            for k = 1:num_feature_blocks
                hf{1,1,k} = zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 sample_dim(k)], 'like', params.data_type_complex);
                hf_t{1,1,k} = hf{1,1,k};
            end
        else
            CG_opts.maxit = params.CG_iter;
            
            % Update the approximate average sample energy using the learning
            % rate. This is only used to construct the preconditioner.
            sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
            sample_energy_t = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy_t, new_sample_energy_t, 'uniformoutput', false);
        end
        
        % Do training
        if seq.frame == 1 && params.update_projection_matrix
            if params.debug
                projection_matrix_init = projection_matrix;
            end
            
            % Initial Gauss-Newton optimization of the filter and
            % projection matrix.
            if params.use_gpu
                [hf, projection_matrix, res_norms] = train_joint_gpu(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            else
                [hf, projection_matrix, res_norms] = train_joint(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
                [hf_t, projection_matrix_t, res_norms_t] = train_joint(hf_t, projection_matrix_t, xlf_t, yf, reg_filter, sample_energy_t, reg_energy, proj_energy_t, params, init_CG_opts);
            end
            
            % Re-project and insert training sample
            xlf_proj = project_sample(xlf, projection_matrix);
            xlf_proj_t = project_sample(xlf_t, projection_matrix_t);
            for k = 1:num_feature_blocks
                if params.use_gpu
                    samplesf{k}(:,:,:,1) = xlf_proj{k};
                else
                    samplesf{k}(1,:,:,:) = permute(xlf_proj{k}, [4 3 1 2]);
                    samplesf_t{k}(1,:,:,:) = permute(xlf_proj_t{k}, [4 3 1 2]);
                end
            end
            
            % Update the gram matrix since the sample has changed
            if strcmp(params.distance_matrix_update_type, 'exact')
                % Find the norm of the reprojected sample
                new_train_sample_norm =  0;
                new_train_sample_norm_t = 0;
                
                for k = 1:num_feature_blocks
                    new_train_sample_norm = new_train_sample_norm + real(gather(2*(xlf_proj{k}(:)' * xlf_proj{k}(:))));% - reshape(xlf_proj{k}(:,end,:,:), [], 1, 1)' * reshape(xlf_proj{k}(:,end,:,:), [], 1, 1));
                    new_train_sample_norm_t = new_train_sample_norm_t + real(gather(2*(xlf_proj_t{k}(:)' * xlf_proj_t{k}(:))));
                end
                
                gram_matrix(1,1) = new_train_sample_norm;
                gram_matrix_t(1,1) = new_train_sample_norm_t;
            end
            
            if params.debug
                norm_proj_mat_init = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix_init)));
                norm_proj_mat = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix)));
                norm_proj_mat_change = sqrt(sum(cellfun(@(P,P2) gather(norm(P(:) - P2(:))^2), projection_matrix_init, projection_matrix)));
                fprintf('Norm init: %f, Norm final: %f, Matrix change: %f\n', norm_proj_mat_init, norm_proj_mat, norm_proj_mat_change / norm_proj_mat_init);
            end
        else
            % Do Conjugate gradient optimization of the filter
            
            if weight_rgb_non>0.3 || PSR_rgb > 7
                [hf, res_norms, CG_state] = train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            end
            if weight_t_non>0.3  || PSR_t > 7
                [hf_t, res_norms_t, CG_state_t] = train_filter(hf_t, samplesf_t, yf, reg_filter, sample_weights_t, sample_energy_t, reg_energy, params, CG_opts, CG_state_t);
            end
            
        end
        
        % Reconstruct the full Fourier series
        hf_full = full_fourier_coeff(hf);
        hf_full_t = full_fourier_coeff(hf_t);
        
        frames_since_last_train = 0;
    else
        frames_since_last_train = frames_since_last_train+1;
    end
    
    % Update the scale filter
    if nScales > 0 && params.use_scale_filter
        scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % debug visualization
    if params.debug
        figure(20)
        subplot_cols = num_feature_blocks;
        subplot_rows = 3;
        for disp_layer = 1:num_feature_blocks;
            subplot(subplot_rows,subplot_cols,disp_layer);
            imagesc(mean(abs(sample_fs(conj(hf_full{disp_layer}))), 3));
            colorbar;
            axis image;
            subplot(subplot_rows,subplot_cols,disp_layer+subplot_cols);
            imagesc(mean(abs(xl{disp_layer}), 3));
            colorbar;
            axis image;
            if seq.frame > 1
                subplot(subplot_rows,subplot_cols,disp_layer+2*subplot_cols);
                imagesc(fftshift(sample_fs(scores_fs_feat{disp_layer}(:,:,1,scale_ind))));
                colorbar;
                axis image;
            end
        end
        
        if train_tracker
            residuals_pcg = [residuals_pcg; res_norms];
            residuals_pcg = [residuals_pcg; res_norms_t];
            res_start_ind = max(1, length(residuals_pcg)-300);
            figure(99);plot(res_start_ind:length(residuals_pcg), residuals_pcg(res_start_ind:end));
            axis([res_start_ind, length(residuals_pcg), 0, min(max(residuals_pcg(res_start_ind:end)), 0.2)]);
        end
    end
    rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
    Box_LarHer((seq.frame+1)/2,:) = rect_position_vis; 
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        Box_LarHer((seq.frame+1)/2,:) = rect_position_vis;  
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
            %             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            % Do visualization of the sampled confidence scores overlayed
            resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            % To visualize the continuous scores, sample them 10 times more
            % dense than output_sz.
            sampled_scores_display = fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz));
            
            figure(fig_handle);
            %                 set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
            alpha(resp_handle, 0.1);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, strcat('#',int2str((seq.frame+1)/2),'/', int2str(seq.len)), 'color', [0 1 1]);
            hold off;
        end  
        drawnow     
    end
end


[seq, results] = get_sequence_results(seq);
disp(['fps: ' num2str(results.fps)])

