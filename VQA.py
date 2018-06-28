import json
import cv2
import string
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import sys

SEQUENCE_LENGTH = 14
BATCH_SIZE = 64
LSTM_SIZE = 512

phase = "train"
with open("./coco/v2_OpenEnded_mscoco_"+phase+"2014_questions.json") as f:
    question_json = json.load(f)
with open("./coco/v2_mscoco_"+phase+"2014_annotations.json") as f:
    answers_json = json.load(f)
with open("./coco/v2_mscoco_"+phase+"2014_complementary_pairs.json") as f:
    pairs_json = json.load(f)
with open("./coco/annotations/captions_"+phase+"2014.json") as f:
    captions_json = json.load(f)

def preprocess_question(question):
    return str(question.lower()).translate(None, string.punctuation)

id_to_file_name = {}
for entry in captions_json["images"]:
    id_to_file_name[entry["id"]] = entry["file_name"]

questions_data = {}
for entry in question_json["questions"]:
    if entry["image_id"] in id_to_file_name:
        questions_data[entry["question_id"]] = {}
        questions_data[entry["question_id"]]["image_id"] = entry["image_id"]
        questions_data[entry["question_id"]]["question"] = preprocess_question(entry["question"])

answer_to_id = {}
id_to_answer = {}
answer_to_freq = {}
yes_no_list = set()
number_list = set()
other_list = set()
answer_type = {}

count  = 0
for entry in answers_json["annotations"]:
    if entry["question_id"] in questions_data:
        questions_data[entry["question_id"]]["answers"] = entry["answers"]
        for answer in entry["answers"]:
            if answer["answer"] not in answer_to_freq:
                answer_to_freq[answer["answer"]] = 1
            else:
                answer_to_freq[answer["answer"]] += 1
            answer_type[answer["answer"]] = entry["answer_type"]

for i in answer_to_freq.keys():
    if answer_to_freq[i] >= 269:
        answer_to_id[i] = count
        id_to_answer[count] = i
        if answer_type[i] == "yes/no":
            yes_no_list.add(count)
        if answer_type[i] == "number":
            number_list.add(count)
        if answer_type[i] == "other":
            other_list.add(count)
        count += 1
NUM_CLASSES = len(id_to_answer.keys())

for question_id in questions_data.keys():
    y = np.zeros(NUM_CLASSES)
    count = 0
    for answer in questions_data[question_id]["answers"]:
        if answer["answer"] in answer_to_id:
            y[answer_to_id[answer["answer"]]] += 1
    y = np.divide(y,10)
    for i in range(len(y)):
        if y[i] >= 0.3:
            y[i] = 1.0
    questions_data[question_id]["y"] = y

question_vocab = set()
for question_id in questions_data.keys():
    question = questions_data[question_id]["question"]
    for word in question.split():
        question_vocab.add(word)

GLOVE_PATH = "./glove.840B.300d.txt"
GLOVE_DIM = 300
glove_embeddings = []
word_to_index = {}
index_to_word = {}
count = 0
with open(GLOVE_PATH) as f:
    for line in f:
        entry = line.split(" ")
        word = entry[0].lower()
        if word in question_vocab and word not in word_to_index:
            glove_embeddings.append(list(map(float, entry[1:])))
            word_to_index[word] = count
            index_to_word[count] = word
            count += 1

word_to_index["<end_token>"] = count
index_to_word[count] = "<end_token>"
glove_embeddings.append([1.0] * GLOVE_DIM)
count += 1

word_to_index["<unk_token>"] = count
index_to_word[count] = "<unk_token>"
count += 1
glove_embeddings.append([0.0] * GLOVE_DIM)

def encode_question(question):
    encoding = []
    count = 0
    for token in question.split():
        if count < SEQUENCE_LENGTH-1:
            try:
                encoding.append([word_to_index[token]])
            except:
                encoding.append([word_to_index["<unk_token>"]])
            count += 1
    while count < SEQUENCE_LENGTH:
        encoding.append([word_to_index["<end_token>"]])
        count += 1
    return encoding

for question_id in questions_data.keys():
    questions_data[question_id]["question_encoding"] = encode_question(questions_data[question_id]["question"])

dataset = []
unique = set()
for pair in pairs_json:
    unique.add(pair[0])
    unique.add(pair[1])
    dataset.append((pair[0], pair[1]))
prev = -1
for question_id in questions_data.keys():
    if question_id not in unique:
        if prev == -1:
            prev = question_id
        else:
            dataset.append((prev, question_id))
            prev = -1
dataset_train = dataset

input_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
input_question = tf.placeholder(tf.int32, [SEQUENCE_LENGTH, BATCH_SIZE])
targets = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
lr = tf.placeholder(tf.float32)
train = tf.placeholder(tf.bool)
preprocess_img = tf.keras.applications.resnet50.preprocess_input(input_img)
resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',input_tensor=preprocess_img,input_shape=(224,224,3))
model_intermediate = tf.keras.Model(inputs=resnet.input, outputs=resnet.get_layer('activation_49').output)
LAYER_SIZE = 7*7
LAYER_DEPTH = 2048

intermediate_features = tf.reshape(model_intermediate.outputs[0], [BATCH_SIZE,LAYER_SIZE,LAYER_DEPTH])
img_features = tf.reshape(resnet.outputs[0], [BATCH_SIZE, 2048])
img_norm = tf.nn.l2_normalize(img_features)
img_embedding = tf.layers.dense(img_norm, 2048, activation=tf.nn.relu)

embedding_matrix = tf.constant(glove_embeddings)
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, input_question)

lstm_1 = tf.contrib.rnn.LSTMCell(LSTM_SIZE, activation=tf.nn.tanh)
lstm_2 = tf.contrib.rnn.LSTMCell(LSTM_SIZE, activation=tf.nn.tanh)

state_1 = lstm_1.zero_state(BATCH_SIZE, tf.float32)
state_2 = lstm_2.zero_state(BATCH_SIZE, tf.float32)

input_sequence = tf.unstack(word_embeddings)
for input_batch in input_sequence:
    word_transform = tf.layers.dense(input_batch, GLOVE_DIM, name="word_input_fc", activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
    lstm_output_1, state_1 = lstm_1(word_transform, state_1)
    lstm_output_2, state_2 = lstm_2(lstm_output_1, state_2)

lstm_rep_1 = tf.concat([state_1[0], state_1[1]], 1)
lstm_rep_2 = tf.concat([state_2[0], state_2[1]], 1)
lstm_rep = tf.concat([lstm_rep_1, lstm_rep_2], 1)

question_fc = tf.layers.dense(lstm_rep, 2048, activation=tf.nn.relu)

pre_joint_rep = tf.multiply(question_fc, img_embedding)

attention_fc1 = tf.layers.dropout(tf.layers.dense(pre_joint_rep, 1024, activation=tf.nn.relu), training=train)
attention_fc2 = tf.layers.dense(attention_fc1, 1024, activation=tf.nn.relu)
attention_logits = tf.layers.dense(attention_fc2, LAYER_SIZE)
attention_probs = tf.reshape(tf.nn.softmax(attention_logits), [BATCH_SIZE, 1, LAYER_SIZE])
attention_features = tf.reshape(tf.matmul(attention_probs, intermediate_features), [BATCH_SIZE, LAYER_DEPTH])
attention_features_embedding= tf.layers.dense(tf.nn.l2_normalize(attention_features), 2048, activation=tf.nn.relu)

joint_rep = tf.multiply(question_fc, attention_features_embedding)
fc1 = tf.layers.dropout(tf.layers.dense(joint_rep, 1024, activation=tf.nn.relu), training=train)
fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
logit = tf.layers.dense(fc2, NUM_CLASSES)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets,logits=logit))

t_vars = []
for i in tf.trainable_variables():
    if "branch" not in i.name and "conv" not in i.name:
        t_vars.append(i)

update = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=t_vars)
output = tf.argmax(logit, axis=1)

saver = tf.train.Saver(max_to_keep=100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


phase = "val"
with open("./coco/v2_OpenEnded_mscoco_"+phase+"2014_questions.json") as f:
    validation_question_json = json.load(f)
with open("./coco/v2_mscoco_"+phase+"2014_annotations.json") as f:
    validation_answers_json = json.load(f)
with open("./coco/annotations/captions_"+phase+"2014.json") as f:
    validation_captions_json = json.load(f)

validation_id_to_file_name = {}
for entry in validation_captions_json["images"]:
    validation_id_to_file_name[entry["id"]] = entry["file_name"]

validation_questions_data = {}
for entry in validation_question_json["questions"]:
    if entry["image_id"] in validation_id_to_file_name:
        validation_questions_data[entry["question_id"]] = {}
        validation_questions_data[entry["question_id"]]["image_id"] = entry["image_id"]
        validation_questions_data[entry["question_id"]]["question"] = preprocess_question(entry["question"])

count = 0
zero_shot = 0
for entry in validation_answers_json["annotations"]:
    if entry["question_id"] in validation_questions_data:
        validation_questions_data[entry["question_id"]]["answer"] = entry["multiple_choice_answer"]
        validation_questions_data[entry["question_id"]]["answers"] = entry["answers"]
        validation_questions_data[entry["question_id"]]["answer_type"] = entry["answer_type"]

for question_id in validation_questions_data.keys():
    validation_questions_data[question_id]["question_encoding"] = encode_question(validation_questions_data[question_id]["question"])

images_to_load = []
for i in dataset_train:
    images_to_load += [questions_data[i[0]]["image_id"], questions_data[i[1]]["image_id"]]
id_to_image = {}
for image_id in images_to_load:
    if image_id not in id_to_image:
        id_to_image[image_id] = cv2.cvtColor(cv2.resize(cv2.imread("./coco/train2014/"+id_to_file_name[image_id]), (224,224)),cv2.COLOR_BGR2RGB)

validation_dataset = validation_questions_data.keys()
images_to_load = []
for i in validation_dataset:
    images_to_load += [validation_questions_data[i]["image_id"]]
validation_id_to_image = {}
for image_id in images_to_load:
    if image_id not in validation_id_to_image:
        validation_id_to_image[image_id] = cv2.cvtColor(cv2.resize(cv2.imread("./coco/val2014/"+validation_id_to_file_name[image_id]), (224,224)),cv2.COLOR_BGR2RGB)

def validate(VALIDATION_EPOCH,num_examples = 12800):
    if VALIDATION_EPOCH != 0:
        saver.restore(sess, "./vqacheckpoints/model"+str(VALIDATION_EPOCH)+".ckpt")
    random.shuffle(validation_dataset)
    index = 0
    total = 0
    correct = 0
    zero_shot = 0
    yes_no_correct = 0
    number_correct = 0
    other_correct = 0
    yes_no_count = 0
    number_count = 0
    other_count = 0
    while index + BATCH_SIZE <= num_examples:
        x_image = []
        x_question = []
        y = []
        ans_type = []
        for q_id in validation_dataset[index:index+BATCH_SIZE]:
            if len(x_image) == 0:
                x_image = [validation_id_to_image[validation_questions_data[q_id]["image_id"]]]
                x_question = np.array(validation_questions_data[q_id]["question_encoding"])
            else:
                x_image += [validation_id_to_image[validation_questions_data[q_id]["image_id"]]]
                x_question = np.hstack((x_question, np.array(validation_questions_data[q_id]["question_encoding"])))
            y.append(validation_questions_data[q_id]["answers"])
            ans_type.append(validation_questions_data[q_id]["answer_type"])
        x_image = np.array(x_image)
        index += BATCH_SIZE
        outp = sess.run(logit, feed_dict={input_img: x_image, input_question: x_question, train: False})
        for i in range(BATCH_SIZE):
            probs = outp[i]
            answer_id = np.argmax(probs)
            if answer_id in id_to_answer:
                answer = id_to_answer[answer_id]
            else:
                answer = ""
            sub_count = 0
            for j in y[i]:
                if j["answer"] == answer:
                    sub_count += 1
            acc = min(sub_count/3.0,1)
            # if sub_count >= 1:
            #     acc = 1
            # else:
            #     acc = 0
            if ans_type[i] == "yes/no":
                yes_no_correct += acc
                yes_no_count += 1
            if ans_type[i] == "other":
                other_correct += acc
                other_count += 1
            if ans_type[i] == "number":
                number_correct += acc
                number_count += 1
            correct += acc
            total += 1
    print "All= ",correct/float(total), "Yes/No =", yes_no_correct/float(yes_no_count), "number =", number_correct/float(number_count),  "other =", other_correct/float(other_count)
    return correct/float(total)

NUM_EPOCHS = 25
RESUME_EPOCH = 0
if RESUME_EPOCH != 0:
    saver.restore(sess, "./vqacheckpoints/model"+str(RESUME_EPOCH)+".ckpt")
    validate(RESUME_EPOCH, len(validation_dataset))
for epoch in range(RESUME_EPOCH+1,NUM_EPOCHS+1):
    num_examples = len(dataset_train)
    random.shuffle(dataset_train)
    index = 0
    while index + (BATCH_SIZE/2) <= num_examples:
        x_image = []
        x_question = []
        y = []
        for q_id1, q_id2 in dataset_train[index:index+BATCH_SIZE/2]:
            if len(x_image) == 0:
                x_image = [id_to_image[questions_data[q_id1]["image_id"]], id_to_image[questions_data[q_id2]["image_id"]]]
                x_question = np.hstack((questions_data[q_id1]["question_encoding"], questions_data[q_id2]["question_encoding"]))
            else:
                x_image += [id_to_image[questions_data[q_id1]["image_id"]], id_to_image[questions_data[q_id2]["image_id"]]]
                x_question = np.hstack((x_question, np.hstack((questions_data[q_id1]["question_encoding"], questions_data[q_id2]["question_encoding"]))))
            y.append(questions_data[q_id1]["y"])
            y.append(questions_data[q_id2]["y"])
        x_image = np.array(x_image)
        index += BATCH_SIZE/2
        _, loss_ = sess.run([update, loss], feed_dict={input_img: x_image,
                                                     input_question: x_question,
                                                     targets: y,
                                                     lr: 0.0001,
                                                      train: True})
        if index%6400 == 0:
            print "epoch=", epoch,"index=", index,"loss=",loss_
    save_path = saver.save(sess, "./vqacheckpoints/model"+str(epoch)+".ckpt")
    print validate(epoch, len(validation_dataset))
