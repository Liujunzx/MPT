
% This demo script runs the ECO tracker with hand-crafted features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
base_path  = './RGBT-Video';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  run video sequence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
video = choose_video(base_path);
video_path = [base_path '/' video];
[seq, ground_truth] = load_video_RGBT(video_path,video);
results = testing_ECO_HC(seq,video);


