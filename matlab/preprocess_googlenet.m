netCNN = googlenet;

test_ds = imageDatastore('train', 'IncludeSubfolders', true,'FileExtensions','.mp4','LabelSource', 'foldername');
files = test_ds.Files;
labels = test_ds.Labels;

idx = 1;
filename = string(files(idx));
video = readVideo(filename);
size(video)
labels(idx)

numFrames = size(video,4);
figure
for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    drawnow
end

inputSize = netCNN.Layers(1).InputSize(1:2);
layerName = "pool5-7x7_s1";

tempFile = fullfile(tempdir,"hmdb51_org.mat");

% if exist(tempFile,'file')
%     load(tempFile,"sequences")
% else
    numFiles = numel(files);
    sequences = cell(numFiles,1);
    
    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles)
        
        video = readVideo(files(i));
        video = centerCrop(video,inputSize);
        
        sequences{i,1} = activations(netCNN,video,layerName,'OutputAs','columns');
    end
    
    save(tempFile,"sequences","-v7.3");
% end

numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.9 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = labels(idxValidation);

numObservationsTrain = numel(sequencesTrain);
sequenceLengths = zeros(1,numObservationsTrain);

for i = 1:numObservationsTrain
    sequence = sequencesTrain{i};
    sequenceLengths(i) = size(sequence,2);
end













