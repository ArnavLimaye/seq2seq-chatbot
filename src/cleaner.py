import re


def replaceContractedByExpandedWords(text):
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"i ain\'t", "i am not", text)
    text = re.sub(r"we ain\'t", "we are not", text)
    # generic
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # non-alphanumeric
    text = re.sub("[^a-zA-Z0-9_ ]", "", text)
    return text


def cleanData(data):
    lowerCaseText = data.lower()
    cleanText = replaceContractedByExpandedWords(lowerCaseText)
    return cleanText


def cleanDataList(anyList):
    cleanList = []
    for item in anyList:
        cleanList.append(cleanData(item))
    return cleanList
