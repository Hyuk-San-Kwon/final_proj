filename = "안녕하세요_31.mp4";
video = readVideo(filename);

video = centerCrop(video,inputSize);
YPred = classify(net,{video});

fprintf("%s\n", YPred);