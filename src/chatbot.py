import os
import constants
from cleaner import cleanDataList, cleanData
from dataReader import getTrainingData
import time
import re
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


######Data Preprocessing######


def mapLineIdsToLines(lines):
    lineIdToLineMap = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if(len(_line)) == 5:
            lineIdToLineMap[_line[0]] = _line[4]
    return lineIdToLineMap


def getConversationsIds(conversations):
    conversationsIds = []
    for conversation in conversations:
        _conversationWithoutMetadata = conversation.split(
            ' +++$+++ ')[-1][1:-1]
        _conversation = _conversationWithoutMetadata.replace(
            "'", "").replace(" ", "")
        conversationsIds.append(_conversation.split(','))
    return conversationsIds


def mapWordsToCount(listOfStrings, wordToCountMap):
    for string in listOfStrings:
        for word in string.split():
            if word not in wordToCountMap:
                wordToCountMap[word] = 1
            else:
                wordToCountMap[word] += 1


def vectorizeStringsList(listOfStrings, wordToIntMap):
    vectorizedStringsList = []
    for string in listOfStrings:
        ints = []
        for word in string.split():
            if word not in wordToIntMap:
                ints.append(wordToIntMap['<OUT>'])
            else:
                ints.append(wordToIntMap[word])
        vectorizedStringsList.append(ints)
    return vectorizedStringsList


lines = getTrainingData(constants.MovieLinesFilePath, constants.FileEncoding)

conversations = getTrainingData(
    constants.MovieConversationsFilePath, constants.FileEncoding)
lineIdToLineMap = mapLineIdsToLines(lines)
# print(lineIdToLineMap['L194'])
conversationsIds = getConversationsIds(conversations)
# print(conversationsIds[0:2])

questions = []
answers = []

for conversationIds in conversationsIds:
    for i in range(len(conversationIds) - 1):
        questions.append(lineIdToLineMap[conversationIds[i]])
        answers.append(lineIdToLineMap[conversationIds[i+1]])

# print(questions[0])
# print(answers[0])

cleanQuestions = cleanDataList(questions)

cleanAnswers = cleanDataList(answers)

# print(cleanQuestions[0])
# print(cleanAnswers[0])

wordToCountMap = {}
mapWordsToCount(cleanQuestions, wordToCountMap)
mapWordsToCount(cleanAnswers, wordToCountMap)

##  May be removed after using tokenizer from Tensorflow ##
threshold = 20
questionsWordsToIntMap = {}
wordNumber = 0
for word, count in wordToCountMap.items():
    if count > threshold:
        questionsWordsToIntMap[word] = wordNumber
        wordNumber += 1

answersWordsToIntMap = {}
wordNumber = 0
for word, count in wordToCountMap.items():
    if count > threshold:
        answersWordsToIntMap[word] = wordNumber
        wordNumber += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionsWordsToIntMap[token] = len(questionsWordsToIntMap) + 1

for token in tokens:
    answersWordsToIntMap[token] = len(answersWordsToIntMap) + 1

# Get reverse dictionary of answerWordToInt
answersIntToWordMap = {w_i: w for w, w_i in answersWordsToIntMap.items()}

# adding EOS at the end of each cleanAnswer
for i in range(len(cleanAnswers)):
    cleanAnswers[i] += ' <EOS>'

questionsVectors = vectorizeStringsList(cleanQuestions, questionsWordsToIntMap)
# print(questionsVectors[0])

answersVectors = vectorizeStringsList(cleanAnswers, answersWordsToIntMap)
# print(answersVectors[0])

# Sorting Q&A by the length
sortedCleanQuestions = []
sortedCleanAnswers = []
for length in range(1, 25 + 1):
    for i in enumerate(questionsVectors):
        if(len(i[1])) == length:
            sortedCleanQuestions.append(questionsVectors[i[0]])
            sortedCleanAnswers.append(answersVectors[i[0]])
# print(sortedCleanAnswers[0])
# print(sortedCleanQuestions[0])

##### Building Seq2Seq model #####


def modelInputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learningRate')
    keepProb = tf.placeholder(tf.float32, name='keepProb')
    return inputs, targets, lr, keepProb


def preProcessTargets(targets, wordToIntMap, batchSize):
    leftSide = tf.fill([batchSize, 1], wordToIntMap['<SOS>'])
    rightSide = tf.strided_slice(targets, [0, 0], [batchSize, -1], [1, 1])
    preProcessedTargets = tf.concat([leftSide, rightSide], 1)
    return preProcessedTargets


def createEncoderRnn(rnnInputs, rnnSize, rnnLayersCount, keepProb, sequenceLength):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keepProb)
    encoderCell = tf.contrib.rnn.MultiRNNCell([lstmDropout] * rnnLayersCount)
    _, encoderState = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoderCell, cell_bw=encoderCell, sequence_length=sequenceLength, inputs=rnnInputs, dtype=tf.float32)
    return encoderState


def decodeTrainingSet(encoderState, decoderCell, decoderEmbeddedInput, sequenceLength, decodingScope, outputFunction, keepProb, batchSize):
    attentionStates = tf.zeros([batchSize, 1, decoderCell.output_size])
    attentionKeys, attentionValues, attentionScoreFunction, attentionConstructFunction = tf.contrib.seq2seq.prepare_attention(
        attentionStates, attention_option='bahdanau', num_units=decoderCell.output_size)
    trainingDecoderFunction = tf.contrib.seq2seq.attention_decoder_fn_train(
        encoderState[0], attentionKeys, attentionValues, attentionScoreFunction, attentionConstructFunction, name='attn_dec_train')
    decoderOutput, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoderCell, trainingDecoderFunction, decoderEmbeddedInput, sequenceLength, scope=decodingScope)
    decoderOutputDropout = tf.nn.dropout(decoderOutput, keepProb)
    return outputFunction(decoderOutputDropout)


def decodeTestSet(encoderState, decoderCell, decoderEmbeddingsMatrix, sosId, eosId, maximumLength, wordsCount, decodingScope, outputFunction, keepProb, batchSize):
    attentionStates = tf.zeros([batchSize, 1, decoderCell.output_size])
    attentionKeys, attentionValues, attentionScoreFunction, attentionConstructFunction = tf.contrib.seq2seq.prepare_attention(
        attentionStates, attention_option='bahdanau', num_units=decoderCell.output_size)
    testDecoderFunction = tf.contrib.seq2seq.attention_decoder_fn_inference(outputFunction,
                                                                            encoderState[0], attentionKeys, attentionValues, attentionScoreFunction,
                                                                            attentionConstructFunction, decoderEmbeddingsMatrix, sosId, eosId,
                                                                            maximumLength, wordsCount, name='attn_dec_inf')
    testPredictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoderCell,
                                                                   testDecoderFunction,
                                                                   scope=decodingScope)
    return testPredictions


def createDecoderRnn(decoderEmbeddedInput, decoderEmbeddingsMatrix, encoderState, wordsCount, sequenceLength, rnnSize, rnnLayersCount, wordToIntMap, keepProb, batchSize):
    with tf.variable_scope("decoding") as decodingScope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
        lstmDropout = tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob=keepProb)
        decoderCell = tf.contrib.rnn.MultiRNNCell(
            [lstmDropout] * rnnLayersCount)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()

        def outputFunction(x): return tf.contrib.layers.fully_connected(x,
                                                                        wordsCount,
                                                                        None,
                                                                        scope=decodingScope,
                                                                        weights_initializer=weights,
                                                                        biases_initializer=biases)
        trainingPredictions = decodeTrainingSet(
            encoderState, decoderCell, decoderEmbeddedInput, sequenceLength, decodingScope, outputFunction, keepProb, batchSize)
        decodingScope.reuse_variables()
        testPredictions = decodeTestSet(encoderState, decoderCell, decoderEmbeddingsMatrix,
                                        wordToIntMap['<SOS>'], wordToIntMap['<EOS>'],
                                        sequenceLength - 1, wordsCount, decodingScope, outputFunction, keepProb, batchSize)
    return trainingPredictions, testPredictions

# Building seq2seq model #


def createSeq2seqModel(inputs, targets, keepProb, batchSize, sequenceLength, totalWordsInAnswers, totalWordsInQuestions, encoderEmbeddingSize, decoderEmbeddingSize, rnnSize, rnnLayersCount, questionsWordsToIntMap):
    encoderEmbeddedInput = tf.contrib.layers.embed_sequence(
        inputs, totalWordsInAnswers + 1, encoderEmbeddingSize, initializer=tf.random_uniform_initializer(0, 1))
    encoderState = createEncoderRnn(
        encoderEmbeddedInput, rnnSize, rnnLayersCount, keepProb, sequenceLength)
    preProcessedTargets = preProcessTargets(
        targets, questionsWordsToIntMap, batchSize)
    decoderEmbeddingsMatrix = tf.Variable(tf.random_uniform(
        [totalWordsInQuestions + 1, decoderEmbeddingSize], 0, 1))
    decoderEmbeddedInput = tf.nn.embedding_lookup(
        decoderEmbeddingsMatrix, preProcessedTargets)
    trainingPredictions, testPredictions = createDecoderRnn(decoderEmbeddedInput, decoderEmbeddingsMatrix, encoderState,
                                                            totalWordsInQuestions, sequenceLength, rnnSize, rnnLayersCount,
                                                            questionsWordsToIntMap, keepProb, batchSize)
    return trainingPredictions, testPredictions


#### Training the Seq2Seq model ####

# Setting the hyper parameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = modelInputs()

# setting sequence length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = createSeq2seqModel(tf.reverse(inputs, [-1]),
                                                            targets,
                                                            keep_prob,
                                                            batch_size,
                                                            sequence_length,
                                                            len(answersWordsToIntMap),
                                                            len(questionsWordsToIntMap),
                                                            encoding_embedding_size,
                                                            decoding_embedding_size,
                                                            rnn_size,
                                                            num_layers,
                                                            questionsWordsToIntMap)

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable)
                         for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token


def apply_padding(batch_of_sequences, wordToIntMap):
    max_sequence_length = max([len(sequence)
                               for sequence in batch_of_sequences])
    return [sequence + [wordToIntMap['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers


def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index: start_index + batch_size]
        answers_in_batch = answers[start_index: start_index + batch_size]
        padded_questions_in_batch = np.array(
            apply_padding(questions_in_batch, questionsWordsToIntMap))
        padded_answers_in_batch = np.array(
            apply_padding(answers_in_batch, answersWordsToIntMap))
        yield padded_questions_in_batch, padded_answers_in_batch


training_validation_split_index = int(len(sortedCleanQuestions) * 0.15)
training_questions = sortedCleanQuestions[training_validation_split_index:]
training_answers = sortedCleanAnswers[training_validation_split_index:]
validation_questions = sortedCleanQuestions[:training_validation_split_index]
validation_answers = sortedCleanAnswers[:training_validation_split_index]

# training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = (
    (len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs+1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {
                                                   inputs: padded_questions_in_batch,
                                                   targets: padded_answers_in_batch,
                                                   lr: learning_rate,
                                                   sequence_length: padded_answers_in_batch.shape[1],
                                                   keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch : {:>4}/{}, Training Loss Error : {:>6.3f}, Training time on 100 batches: {:d} seconds'.format(epoch, epochs,
                                                                                                                                         batch_index,
                                                                                                                                         len(
                                                                                                                                             training_questions)//batch_size,
                                                                                                                                         total_training_loss_error/batch_index_check_training_loss,
                                                                                                                                         int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / \
                (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now !!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry I do not speak better, I need to practice more. ')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print('My apologies . I cannot speak better anymore. This is the best I can do. ')
print('Game Over')


#### Testing the Seq2Seq Model ####

# Loading the weights and running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Converting the questions from strings to lists of encoding integers


def convert_stringToInt(question, wordToIntMap):
    question = cleanData(question)
    return [wordToIntMap.get(word, wordToIntMap['<OUT>']) for word in question.split()]


# Setting up the chatbot
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_stringToInt(question, questionsWordsToIntMap)
    question = question + \
        [questionsWordsToIntMap['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(
        test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersIntToWordMap[i] == 'i':
            token = ' I'
        elif answersIntToWordMap[i] == '<EOS>':
            token = '.'
        elif answersIntToWordMap[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersIntToWordMap[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
