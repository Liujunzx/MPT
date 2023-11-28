function [seq, ground_truth] = load_video_GTOT(video_path,sequence_name)

%ground_truth = dlmread([video_path '/init.txt']); %����LasHer
%ground_truth = dlmread([video_path '/visible.txt']); %����RGBT234
ground_truth = dlmread([video_path '/groundtruth.txt']);     %����GTOT
seq.format = 'otb';
seq.len = size(ground_truth, 1);
%seq.init_rect = ground_truth(1,:);   %��ӦRGBT234
  seq.init_rect = [ground_truth(1,1),ground_truth(1,2),...
      ground_truth(1,3)-ground_truth(1,1),ground_truth(1,4)-ground_truth(1,2)];   %����GTOT

img_path = [video_path '/img/'];
if strcmp(sequence_name, 'David')
    start_frame = 300;end_frame = 770;
elseif strcmp(sequence_name, 'Football1')
    start_frame = 1;end_frame = 74;
elseif strcmp(sequence_name, 'Freeman3')
    start_frame = 1;end_frame = 460;
elseif strcmp(sequence_name, 'Freeman4')
    start_frame = 1;end_frame = 283;
else
    %start_frame = 1; end_frame = seq.len;
    start_frame = 1; end_frame = seq.len*2;  %��仰�����ܿɼ��ⵥԴͼ��
    %start_frame = 2; end_frame = seq.len*2;  %��仰�����ܺ���ⵥԴͼ��   
end
    
if strcmp(sequence_name, 'BlurCar1')
    nn = 247;
elseif strcmp(sequence_name, 'BlurCar3')
    nn = 3;
elseif strcmp(sequence_name, 'BlurCar4')
    nn = 18;
else
    nn = 1;
end
    
if exist([img_path num2str(nn, '%04i.png')], 'file'),
    img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.png']);
elseif exist([img_path num2str(nn, '%04i.jpg')], 'file'),
    img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.jpg']);
    %img_files = num2str((start_frame+nn:2:end_frame+nn)', [img_path '%04i.jpg']);
    %����������ƺ�����ɼ���ͼ�����루�л���
elseif exist([img_path num2str(nn, '%04i.bmp')], 'file'),
    img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.bmp']);
elseif exist([img_path num2str(nn, '%05i.jpg')], 'file'),
    img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%05i.jpg']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);
seq.start_frame = start_frame;
seq.end_frame = end_frame;
end

