netCNN = resnet50;
cnnLayers = layerGraph(netCNN);

layerNames = ["fc1000" "fc1000_softmax" "ClassificationLayer_fc1000"];
cnnLayers = removeLayers(cnnLayers, layerNames);

inputSize = netResNet50.Layers(1).InputSize(1:2);
averageImage = netResNet50.Layers(1).Mean;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');

layers = [inputLayer sequenceFoldingLayer('Name','fold')];
lgraph = addLayers(cnnLayers, layers);
lgraph = removeLayers(lgraph, 'input_1');
lgraph = connectLayers(lgraph,"fold/out","conv1");

lstmLayers = netLSTM.Layers;
lstmLayers(1) = [];

layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph, layers);
lgraph = connectLayers(lgraph, "avg_pool", "unfold/in"); 
lgraph = connectLayers(lgraph, "fold/miniBatchSize", "unfold/miniBatchSize");

analyzeNetwork(lgraph)

net = assembleNetwork(lgraph);
