from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["This is a text", "Hello World", "Welcome to this programming tutorial"]

tokenizer = Tokenizer(oov_token="<UNKNOWN>")
tokenizer.fit_on_texts(texts)

texts = tokenizer.texts_to_sequences(texts)

sequences = pad_sequences(texts, maxlen=5, padding='post')

print(sequences)
print()
text = ["That was my text"]
text = tokenizer.texts_to_sequences(text)
print(text)

