# E-Commerce Reviews Analysis with Machine Learning
# By Sangeetha Priya RE

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

window = tk.Tk()
window.title('Sangeetha Priya RE - E-Commerce Review Processing By pairwise ranking and sentiment analysis')
Name = tk.Label(text="Select The DataSet")
Name1 = tk.Label(text="E-Commerce Review Processing By pairwise ranking and sentiment analysis")
Name2 = tk.Label(text="By Sangeetha Priya RE")
Name1.pack()
Name2.pack()
Name.pack()

def select_file():
    global filename
    filename =  0
    filetypes = (
        ('csv files', '*.csv'),
    )

    filename = fd.askopenfilename(
        title='Select a CSV File',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
    )
    showinfo(
        title='Processing...',
        message='Close the "Select The DataSet" Dialog Box to Get Analytics'
    )

    

open_button = ttk.Button(
    window,
    text='Open a File',
    command=select_file
)

open_button.pack(expand=True)


window.mainloop()

while (filename!=0):
    df = pd.read_csv(filename)
    for column in ["Division Name","Department Name","Class Name","Review Text"]:
        df = df[df[column].notnull()]
    df.drop(df.columns[0], inplace=True, axis=1)
    df['Label'] = 0
    df.loc[df.Rating >= 3, ['Label']] = 1
    df['Word Count'] = df['Review Text'].str.split().apply(len)
    df.describe().T.drop('count', axis=1)
    df[['Title', 'Division Name', 'Department Name', 'Class Name']].describe(include=['O']).T.drop('count', axis=1)
    
    ## Average Rating and Recommended IND by Class Name Correlation
    key = 'Class Name'
    temp = (df.groupby(key)[['Rating', 'Recommended IND', 'Age']]
            .aggregate(['count', 'mean']))
    temp.columns = ['Count', 'Rating Mean', 'Recommended Likelihood Count',
                    'Recommended Likelihood', 'Age Count', 'Age Mean']
    temp.drop(['Recommended Likelihood Count', 'Age Count'], axis=1, inplace=True)

    # Plot Correlation Matrix
    f, ax = plt.subplots(figsize=[10, 7])
    ax = sns.heatmap(temp.corr(),
                    annot=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Correlation Coefficient for Mean and Count for\nRating, Recommended Likelihood, and Age\nGrouped by {}'.format(key))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('meanrating-recommended-classname-corr.png', format='png', dpi=300)
    plt.show()
    print('Class Categories:\n',df['Class Name'].unique())
    g = sns.jointplot(y='Recommended Likelihood', x='Age Mean', data=temp,
                    kind='reg', color='b')
    plt.subplots_adjust(top=0.999)
    g.fig.suptitle('Age Mean and Recommended Likelihood\nGrouped by Clothing Class')
    plt.savefig('meanage-recommended-clothing.png', format='png', dpi=300)
    plt.ylim(.7, 1.01)
    # Working with Text
    pd.set_option('max_colwidth', 500)
    df[["Title","Review Text", "Rating"]].sample(7)
    ## Text Cleaning
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import RegexpTokenizer

    ps = PorterStemmer()

    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))

    def preprocessing(data):
        txt = data.str.lower().str.cat(sep=' ') #1
        words = tokenizer.tokenize(txt) #2
        words = [w for w in words if not w in stop_words] #3
        #words = [ps.stem(w) for w in words] #4
        return words
    import matplotlib as mpl

    stopwords = set(STOPWORDS)
    size = (20, 10)

    def cloud(text, title, stopwords=stopwords, size=size):
        mpl.rcParams['figure.figsize'] = (10.0, 10.0)
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['figure.subplot.bottom'] = .1
        
        wordcloud = WordCloud(width=1600, height=800,
                            background_color='black',
                            stopwords=stopwords).generate(str(text))
        
        fig = plt.figure(figsize=size, facecolor='k')
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title(title, fontsize=50, color='y')
        plt.tight_layout()
        plt.savefig('{}.png'.format(title), format='png', dpi=300)
        plt.show()
        
    def wordfreqviz(text, x):
        word_dist = nltk.FreqDist(text)
        top_N = x
        rslt = pd.DataFrame(word_dist.most_common(top_N),
                            columns=['Word', 'Frequency']).set_index('Word')
        mpl.style.use('ggplot')
        rslt.plot.bar(rot=0)
        
    def wordfreq(text, x):
        word_dist = nltk.FreqDist(text)
        top_N = x
        rlst = pd.DataFrame(word_dist.most_common(top_N),
                            columns=['Word', 'Frequency']).set_index('Word')
        return rlst
    new_stop = set(STOPWORDS)
    new_stop.update([x.lower() for x in list(df['Class Name'][df['Class Name'].notnull()].unique())]
                    + ['dress', 'petite'])

    # Cloud
    cloud(text=df.Title[df.Title.notnull()].astype(str).values,
        title='WordCloud for Titles',
        stopwords=new_stop,
        size = (7,4))
    title ='Most Frequent Words in Highly Rated Comments'
    temp = df['Review Text'][df.Rating.astype(int) >= 3]

    # Modify Stopwords to Exclude Class types, suchs as 'dress'
    new_stop = set(STOPWORDS)
    new_stop.update([x.lower() for x in list(df['Class Name'][df['Class Name'].notnull()].unique())]
                    + ['dress', 'petite'])

    # Cloud
    cloud(text= temp.values, title=title,stopwords= new_stop)

    # Bar Chart
    wordfreq(preprocessing(temp), 20).plot.bar(rot=45, legend=False, figsize=(15, 5), color='g',
                                            title=title)
    plt.ylabel('Occurrence Count')
    plt.xlabel('Most Frequent Words')
    plt.tight_layout()
    plt.savefig('most-freq-words-high-rate-comments.png', format='png', dpi=300)
    plt.show()

    # Low Raited
    title ='Most Frequent Words in Low Rated Comments'
    temp = df['Review Text'][df.Rating.astype(int) < 3]

    # Modify Stopwords to Exclude Class types, suchs as 'dress'
    new_stop = set(STOPWORDS)
    new_stop.update([x.lower() for x in list(df['Class Name'][df['Class Name'].notnull()].unique())]
                    + ['dress', 'petite', 'skirt', 'shirt'])

    # Cloud
    cloud(temp.values, title=title, stopwords=new_stop)
    reviews = df['Review Text'].astype(str).str.lower()
    type(reviews)
    features = reviews.tolist()
    features
    import re
    from string import punctuation
    for index in range(len(features)):
        all_text = ''.join([character for character in features[index] if character not in punctuation])
        features[index] = re.split(r'\n|\r', all_text)
        features[index] = ' '.join([word for word in features[index]])
    features
    labels = np.array(df['Recommended IND'], np.int)
    labels.shape
    labels[labels == 1].shape[0]
    labels[labels == 0].shape[0]
    from keras.utils import to_categorical
    labels = to_categorical(labels)
    labels[:10]
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    t = Tokenizer()
    t.fit_on_texts(features)
    vocabulary_size = len(t.word_index) + 1
    print('Vocabulary size : {}'.format(vocabulary_size))
    encoded_features = t.texts_to_sequences(features)

    max_length = 300

    padded_features = pad_sequences(encoded_features, maxlen=max_length, padding='post')
    embeddings_index = dict()
    with open('/home/darth/GitHub Projects/senti-internship-notes/abienagarap/word-vectors/glove.840B.300d.txt') as file:
        data = file.readlines()
        
    # store <key, value> pair of FastText vectors
    for line in data[1:]:
        word, vec = line.split(' ', 1)
        embeddings_index[word] = np.array([float(index) for index in vec.split()], dtype='float32')
    print('Loaded {} word vectors.'.format(len(embeddings_index)))


    embedding_matrix = np.zeros((vocabulary_size, max_length))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    words = []
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            words.append(word)
    print('{} words covered.'.format(len(words)))
    percentage = (len(words) / vocabulary_size) * 100.00
    print('{}% of {} words were covered'.format(percentage, vocabulary_size))
    def train_test_split(features, labels, **kwargs):
        
        # concatenate the features and labels array
        dataset = np.c_[features, labels]

        # shuffle the dataset
        np.random.shuffle(dataset)

        # split the dataset into features, labels
        features, labels = dataset[:, 0:max_length], dataset[:, max_length:]

        # get the split size for training dataset
        split_index = int(kwargs['train_size'] * len(features))

        # split the dataset into training/validation dataset
        train_features, validation_features = features[:split_index], features[split_index:]
        train_labels, validation_labels = labels[:split_index], labels[split_index:]

        # get the split size for validation dataset
        split_index = int(kwargs['validation_size'] * len(validation_features))

        # split the validation dataset into validation/testing dataset
        validation_features, test_features = validation_features[:split_index], validation_features[split_index:]
        validation_labels, test_labels = validation_labels[:split_index], validation_labels[split_index:]

        # return the partitioned dataset
        return [train_features, train_labels], [validation_features, validation_labels], [test_features, test_labels]
    train_dataset, validation_dataset, test_dataset = train_test_split(features=padded_features, labels=labels,
                                                                    train_size=0.60, validation_size=0.50)
    print('Dataset size : {}'.format(padded_features.shape[0]))
    print('Train dataset size : {}'.format(train_dataset[0].shape[0]))
    print('Validation dataset size : {}'.format(validation_dataset[0].shape[0]))
    print('Test dataset size : {}'.format(test_dataset[0].shape[0]))
    from keras import callbacks
    from keras.layers import Bidirectional
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Embedding
    from keras.layers import LSTM
    from keras.models import Sequential
    from sklearn.model_selection import StratifiedKFold
    model = Sequential()
    e = Embedding(vocabulary_size, max_length,
                weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.50))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_dataset[0], train_dataset[1], epochs=32, batch_size=256, verbose=1,
            validation_data=(validation_dataset[0], validation_dataset[1]))

    score = model.evaluate(test_dataset[0], test_dataset[1], verbose=1)

    print('loss : {}, acc : {}'.format(score[0], score[1]))
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    test_predictions = model.predict(test_dataset[0])
    test_predictions = np.argmax(test_predictions, axis=1)

    class_names = ['(0) Not recommended class', '(1) Recommended class']
    report = classification_report(np.argmax(test_dataset[1], axis=1), test_predictions, target_names=class_names)
    print(report)
    conf_matrix = confusion_matrix(np.argmax(test_dataset[1], axis=1), test_predictions)
    print(conf_matrix)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 16}, cmap='coolwarm', fmt='.2f')
    plt.savefig('conf_matrix_recommendation.png', format='png', dpi=300)
    from sklearn.metrics import roc_auc_score

    roc = roc_auc_score(y_score=test_predictions, y_true=np.argmax(test_dataset[1], 1))
    print(roc)
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(np.argmax(test_dataset[1], 1), test_predictions)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, lw=2, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right', fontsize=16)
    plt.savefig('roc.png', format='png', dpi=300)
    plt.show()
    ## Sentiment Classification
    labels = np.array(df['Sentiment'])
    labels
    labels = np.array([2 if label == 'Positive' else (1 if label == 'Neutral' else 0) for label in labels],
                    np.int)
    labels
    positive_class = int(labels[labels == 2].shape[0])
    neutral_class = int(labels[labels == 1].shape[0])
    negative_class = int(labels[labels == 0].shape[0])

    df = pd.DataFrame.from_dict({'positive': [positive_class], 'negative': [negative_class], 'neutral': [neutral_class]})

    plt.figure(figsize=(8, 8))
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    ax = sns.barplot(data=df)
    ax = ax.set_xlabel('Frequency Distribution of Sentiment Classes')
    labels = to_categorical(labels)
    train_dataset, validation_dataset, test_dataset = train_test_split(features=padded_features, labels=labels,
                                                                    train_size=0.60, validation_size=0.50)
    model = Sequential()
    e = Embedding(vocabulary_size, max_length,
                weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(256), merge_mode='sum'))
    model.add(Dropout(0.50))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_dataset[0], train_dataset[1], epochs=32, batch_size=256, verbose=1,
            validation_data=(validation_dataset[0], validation_dataset[1]))

    score = model.evaluate(test_dataset[0], test_dataset[1], verbose=1)

    print('loss : {}, acc : {}'.format(score[0], score[1]))
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    test_predictions = model.predict(test_dataset[0])
    test_predictions = np.argmax(test_predictions, axis=1)

    class_names = ['(0) Negative class', '(1) Neutral class', '(2) Positive class']
    report = classification_report(np.argmax(test_dataset[1], axis=1), test_predictions, target_names=class_names)
    print(report)
    conf_matrix = confusion_matrix(np.argmax(test_dataset[1], axis=1), test_predictions)
    print(conf_matrix)
    plt.figure(figsize=(8, 8))
    plt.savefig('conf_matrix_sentiment.png', format='png', dpi=300)
    sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 16}, cmap='coolwarm', fmt='.2f')