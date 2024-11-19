from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
import tensorflow as tf
from scipy.special import softmax

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Setăm vizibilitatea doar pentru GPU-ul dorit (de exemplu, GPU 1)
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        print("TensorFlow configurat pentru a folosi GPU 1")
    except RuntimeError as e:
        print("Eroare la configurarea TensorFlow pentru GPU:", e)

# Configurarea GPU pentru transformers
# Specificăm GPU-ul 1 pentru torch
if torch.cuda.is_available():
    device = torch.device("cuda:1")  # Alege GPU-ul 1
    print("Torch configurat pentru a folosi GPU 1")
else:
    device = torch.device("cpu")
    print("Torch configurat să folosească CPU")

def analyze_large_text_sentiment(text, model_name='nlptown/bert-base-multilingual-uncased-sentiment', max_length=511):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Split text into chunks of max_length tokens
    encoded_chunks = []
    text_chunks = []
    words = text.split()
    chunk = ""

    for word in words:
        if len(tokenizer(chunk + word)['input_ids']) < max_length:
            chunk += word + " "
        else:
            text_chunks.append(chunk.strip())
            chunk = word + " "

    if chunk:
        text_chunks.append(chunk.strip())

    # Analyze each chunk and accumulate the scores
    aggregated_scores = np.zeros(len(model.config.id2label))

    for chunk in text_chunks:
        encoded_input = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        aggregated_scores += scores

    # Calculate the average scores
    averaged_scores = aggregated_scores / len(text_chunks)

    # Create a dictionary to map labels to their average scores
    labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    sentiment_results = dict(zip(labels, averaged_scores))

    return sentiment_results

# Exemplu de utilizare:
text = """A good parable--like "The Prodigal Son"--should comfort the afflicted and afflict the comfortable. The problem with this little book is that it does precisely the opposite.

Coelho's message--and, boy, is this a book with a message--is that each of us has his own Personal Legend, and that if we recognize that legend and pursue it sincerely, everything in the Universe (which is after all made up--wind, stone, trees--of the same stuff we are) will conspire to help us achieve it. Corollaries: 1) people who don't recognize their legends are never happy, 2) people who fail to realize their legends are afraid, and 3) people who refuse to pursue their legends, even when they know what they are, are both unhappy and afraid. (I admit I've left out a nuance or two here and there, but not many. There aren't more than three or four nuances in the book.)

I fear that the result of taking such a message seriously will be to make the successful even more self-satisfied, the narcissistic more self-absorbed, and the affluent more self-congratulatory. At the same time, those who are unfortunate will blame themselves for their bad fortune, those who lack self-esteem will lose what little they have, and the poor will see--no, not God, as the beatitude says, but--the poor will see they have only themselves to blame.

Perhaps I am being too harsh. I can see how a few individual young persons, hemmed in by parental expectations and seeking their own paths, may find enough hope and courage here to help them venture forth. But I am convinced the damage done by books like this--like The Secret, The Celestine Prophecy, and anything ever written by the late Dr. Wayne Dyer (or, for that matter, anything he may ever choose to channel from beyond the grave)--is far greater than the little good they may achieve.

If you like parables, don't read this book. Go read a book of Hasidic tales collected by Martin Buber, a book of Sufi stories collected by Idries Shah, or a book of parables and sayings by Anthony de Mello instead.

Or then again, you could just try Jesus. Jesus is always good. """
result = analyze_large_text_sentiment(text)
print(result)
