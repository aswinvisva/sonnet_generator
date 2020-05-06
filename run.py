from dataset_gen import Generator
from model import Model
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
import numpy as np

if __name__ == '__main__':
    gen = Generator()
    train, labels, dictionary = gen.generate_data("ern2017-session4-shakespeare-sonnets-dataset.txt")
    model = Model()
    model.build_model()

    trained_model = load_model('lstm_model.h5')
    sentence = "Thy vengeful mother shall meet her doom if she come to my kingdom alone."
    words = word_tokenize(sentence)
    words = [word.lower() for word in words]
    encoded_list = []
    inv_map = {v: k for k, v in dictionary.items()}

    for word in words:
        encoded_list.append(dictionary[word])

    test_data = np.array(encoded_list)
    print(len(encoded_list))

    no_words = 500

    for i in range(no_words):
        prediction = trained_model.predict(test_data.reshape(1, 15))
        # print(prediction)
        prediction = prediction[0]
        # Get the indices of maximum element in numpy array
        result = np.where(prediction == np.amax(prediction))
        index = result[0][0]
        # print(np.amax(prediction))
        print(inv_map[index], end=" ")

        test_data = test_data.tolist()
        test_data = test_data[1:] + [index]
        test_data = np.array(test_data)


    # model.fit_model(train, labels)

