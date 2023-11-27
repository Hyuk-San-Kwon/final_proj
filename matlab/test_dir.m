% 테스트 데이터셋 불러오기
testDatastore = imageDatastore('test', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.mp4', ...
    'LabelSource', 'foldernames');

% 결과와 실제 레이블을 저장할 배열 초기화
YPreds = [];
YTests = [];

% 각 비디오 파일에 대해 분류 수행
for i = 1:numel(testDatastore.Files)
    filename = testDatastore.Files{i};
    video = readVideo(filename);
    video = centerCrop(video, inputSize);
    YPred = classify(net, {video});
    
    % 예측 결과와 실제 레이블 저장
    YPreds = [YPreds; YPred];
    YTests = [YTests; testDatastore.Labels(i)];
end

% 정확도 계산
accuracy = sum(YPreds == YTests) / numel(YTests);

% 결과 출력
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);