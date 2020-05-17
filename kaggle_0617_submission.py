from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
import time
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
# import xgboost as xgb
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

def read_LSTM(reset_data):
	from keras.preprocessing.text import Tokenizer
	from keras.preprocessing import sequence
	import os

	# load data
	if(reset_data or not os.path.exists('lstm_data_vector.npy') or not os.path.exists("lstm_labels_vector.npy") or not os.path.exists("lstm_qid_vector.npy")):
# 		train = pd.read_csv('../input/smaller-quora/small_train.csv')
		train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
		train = train.drop('qid', axis=1)
# 		test = pd.read_csv('../input/smaller-quora/small_test.csv')
		test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
		test_data = test['question_text']
		qid = test.drop('question_text', axis=1)

		training_data = train['question_text']
		training_labels = train['target'].to_numpy()    
		
		
		# DO PREPROCESSING
		# prepare tokenizer
		t = Tokenizer()
		t.fit_on_texts(training_data)
		vocab_size = len(t.word_index) + 1
		print(vocab_size)
		
		# integer encode the questions
		encoded_train = t.texts_to_sequences(training_data)
		encoded_test = t.texts_to_sequences(test_data)
		# print(encoded_train)

		# pad questions to a max length of 50-100 words (idk a paper said average question length was 27)
		max_question_length = 50
		padded_train = sequence.pad_sequences(encoded_train, maxlen=max_question_length, padding='post')
		padded_test = sequence.pad_sequences(encoded_test, maxlen=max_question_length, padding='post')
		# print(padded_train)

		# Save Data
		np.save("lstm_data_vector.npy", padded_train)
		np.save("lstm_labels_vector.npy", training_labels)
		np.save("lstm_qid_vector.npy", qid)
		np.save("lstm_test_vector.npy", padded_test)
		
	else:
		padded_train = np.load("lstm_data_vector.npy")
		training_labels = np.load("lstm_data_vector.npy")
		qid = np.load("lstm_qid_vector.npy")
		padded_test = np.load("lstm_test_vector.npy")



	x_train, x_test, y_train, y_test = train_test_split(padded_train, training_labels, test_size=0.01, random_state=27)
	print('x_train.shape: ', x_train.shape)
	print('x_test.shape: ', x_test.shape)
	print('y_train.shape: ', y_train.shape)
	print('y_test.shape: ', y_test.shape)
	return padded_test, qid, x_train, x_test, y_train, y_test, vocab_size, max_question_length

def do_LSTM(actual_test, x_train, x_test, y_train, y_test, vocab_size, max_question_length):
	from keras.models import Sequential
	from keras.layers import Dense, LSTM
	from keras.layers.embeddings import Embedding

	# define model

	embedding_vecor_length = 50 # num features for each sentence?
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_question_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(x_train, y_train, epochs=3, batch_size=64)
	
	print("starting predict")
	start = time.time()
	predictions = model.predict_classes(x_test)
	print('Time to predict: ', time.time() - start)    
	
	print("starting Real predict")
	start = time.time()
	actual_pred = model.predict_classes(actual_test)
	print('Time to real predict: ', time.time() - start)    
	
	return actual_pred, predictions


def main():
	lstm = True
	reset_data = True
	if lstm:
		actual_test, qid, x_train, x_test, y_train, y_test, vocab_size, max_question_length = read_LSTM(reset_data)
	else:
		return
	print("starting training")
	start = time.time()
	# use actual_test instead of x_test to do real predictions
	# tes = do_LSTM(x_train, x_test, y_train, y_test, vocab_size, max_question_length)
	actual_pred, tes = do_LSTM(actual_test, x_train, x_test, y_train, y_test, vocab_size, max_question_length)
	

	print('Time to train & predict: ', time.time() - start)
	print("confusion matrix:\n", confusion_matrix(y_test, tes))
	print("accuracy: ", accuracy_score(y_test, tes))
	print("f1: ", f1_score(y_test, tes, average='macro'))
	print("classification report:\n", classification_report(y_test, tes))
	print("precision: ", precision_score(y_test, tes))
	print("recall: ", recall_score(y_test, tes))
	
	qid['prediction'] = actual_pred
	qid.to_csv('submission.csv', index=False)

if __name__ == "__main__":
	main()
