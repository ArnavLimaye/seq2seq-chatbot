def getTrainingData(filePath, encoding):
    file = open(filePath, encoding=encoding, errors='ignore')
    fileContent = file.read().split('\n')
    file.close()
    return fileContent
