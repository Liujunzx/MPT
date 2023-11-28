function [seq, im, im_I] = get_sequence_frame(seq)

seq.frame = seq.frame + 2;

if strcmpi(seq.format, 'otb')
    if seq.frame > seq.num_frames
        im = [];
        im_I= [];
    else
        im = imread(seq.image_files{seq.frame});
        im_I = imread(seq.image_files{seq.frame+1});
        if ndims(im_I) == 2
            im_I = repmat(im_I,[1,1,3]);
        end

    end
elseif strcmpi(seq.format, 'vot')
    [seq.handle, image_file] = seq.handle.frame(seq.handle);
    if isempty(image_file)
        im = [];
    else
        im = imread(image_file);
    end
else
    error('Uknown sequence format');
end