<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow as tf
import torch.nn.functional as F

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


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = [""" A good parable--like "The Prodigal Son"--should comfort the afflicted and afflict the comfortable. The problem with this little book is that it does precisely the opposite.

Coelho's message--and, boy, is this a book with a message--is that each of us has his own Personal Legend, and that if we recognize that legend and pursue it sincerely, everything in the Universe (which is after all made up--wind, stone, trees--of the same stuff we are) will conspire to help us achieve it. Corollaries: 1) people who don't recognize their legends are never happy, 2) people who fail to realize their legends are afraid, and 3) people who refuse to pursue their legends, even when they know what they are, are both unhappy and afraid. (I admit I've left out a nuance or two here and there, but not many. There aren't more than three or four nuances in the book.)

I fear that the result of taking such a message seriously will be to make the successful even more self-satisfied, the narcissistic more self-absorbed, and the affluent more self-congratulatory. At the same time, those who are unfortunate will blame themselves for their bad fortune, those who lack self-esteem will lose what little they have, and the poor will see--no, not God, as the beatitude says, but--the poor will see they have only themselves to blame.

Perhaps I am being too harsh. I can see how a few individual young persons, hemmed in by parental expectations and seeking their own paths, may find enough hope and courage here to help them venture forth. But I am convinced the damage done by books like this--like The Secret, The Celestine Prophecy, and anything ever written by the late Dr. Wayne Dyer (or, for that matter, anything he may ever choose to channel from beyond the grave)--is far greater than the little good they may achieve.

If you like parables, don't read this book. Go read a book of Hasidic tales collected by Martin Buber, a book of Sufi stories collected by Idries Shah, or a book of parables and sayings by Anthony de Mello instead.

Or then again, you could just try Jesus. Jesus is always good.""", """ I need to start this review by stating 1) I can't stand self-help books and 2) I'm a feminist (no, I don't hate men- some men are quite awesome, but I am very conscious of women and our place in the world.)

Short summary (mild spoilers): A boy named Santiago follows his 'Personal Legend' in traveling from Spain to the Pyramids in Egypt searching for treasure. Along the way, he learns 'the Language of the World' the 'Soul of the World' and discovers that the 'Soul of God' is 'his own soul.'

If the statements in quotes above ('personal legend', etc) fascinate you, then you'll enjoy this book. If you think they are hokey and silly, then you'll think this is a terrible book. If you think statements such as "When you want something, all the universe conspires you to achieve it" and "All things are one" are moving and life-changing, you'll love this book. If such statements have you rolling your eyes, then this isn't your cup of tea.

Its not that I find anything wrong with these messages. They are important, but must be balanced with responsibility. In my experience, 'following your dreams' (or personal legend) is not the only way toward wisdom and strength. Is the person who struggles to put food on the table every day for his or her family, consciously realizing that he or she may not be following his or her 'personal legend' any less heroic than some traveler who leaves everything and everyone he or she is responsible for to go on a spiritual quest? Coelho comes close to labeling such people, as losers in life, which I find completely off the mark as some of these people have the most to offer in terms of wisdom.

The issue of responsibility is also part of this book's sexism. The main male characters in the novel have 'Personal Legends' - they are either seeking them, or have achieved them, or have failed to achieve them. But Coelho never mentions 'Personal Legend' with regard to women, other than to say that Fatima, Santiago's fiance, is 'a part of Santiago's Personal Legend." Thats fine, but what about her own Personal Legend? Instead of traveling to find her dreams, she is content to sit around, do chores, and stare everyday at the desert to wait for his return. This is her 'fate' as a desert women. The fact that women don't have Personal Legends is even more galling considering the fact that according to Coelho, even minerals such as lead and copper have Personal Legends, allowing them to 'evolve' to something better (ie, gold).

In the ideal world presented in THE ALCHEMIST, it seems that the job of men is to seek out their personal legends, leaving aside thoughts of family and responsibility, and its the job of women to let them, and pine for their return. Of course, someone has to do the unheroic, inconvenient work of taking care of the children, the animals, the elderly, the ill...If everyone simply goes off on spiritual quests, deciding they have no responsibility other than to seek their Personal Legends, no one would be taking responsibility for the unglamorous work that simply has to take place for the world to run.

On the other hand, what if both men and women are allowed to struggle towards their 'Personal Legends,' and help each other as best as they can towards them, but recognize that their responsibilities may force them to defer, compromise, or even 'sacrifice' their dreams? This may seem depressing, but it isn't necessarily. Coelho seems to think that Personal Legends are fixed at childhood (or at birth, or even before) and are not changeable: they have to be followed through to the end, no matter how silly. But in my experience, many people have chosen to adjust, compromise, and even 'give up' on their dreams, only to find that life grants them something better, or they have a new, better dream to follow, a path providing greater wisdom. For me, these people have a more realistic, more humble, more fair, and less cliched vision of the world than Paulo Coelho's vision in THE ALCHEMIST."""]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim = 0).item()

# Exemplu de utilizare
similarity = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])
=======
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow as tf
import torch.nn.functional as F

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


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = [""" A good parable--like "The Prodigal Son"--should comfort the afflicted and afflict the comfortable. The problem with this little book is that it does precisely the opposite.

Coelho's message--and, boy, is this a book with a message--is that each of us has his own Personal Legend, and that if we recognize that legend and pursue it sincerely, everything in the Universe (which is after all made up--wind, stone, trees--of the same stuff we are) will conspire to help us achieve it. Corollaries: 1) people who don't recognize their legends are never happy, 2) people who fail to realize their legends are afraid, and 3) people who refuse to pursue their legends, even when they know what they are, are both unhappy and afraid. (I admit I've left out a nuance or two here and there, but not many. There aren't more than three or four nuances in the book.)

I fear that the result of taking such a message seriously will be to make the successful even more self-satisfied, the narcissistic more self-absorbed, and the affluent more self-congratulatory. At the same time, those who are unfortunate will blame themselves for their bad fortune, those who lack self-esteem will lose what little they have, and the poor will see--no, not God, as the beatitude says, but--the poor will see they have only themselves to blame.

Perhaps I am being too harsh. I can see how a few individual young persons, hemmed in by parental expectations and seeking their own paths, may find enough hope and courage here to help them venture forth. But I am convinced the damage done by books like this--like The Secret, The Celestine Prophecy, and anything ever written by the late Dr. Wayne Dyer (or, for that matter, anything he may ever choose to channel from beyond the grave)--is far greater than the little good they may achieve.

If you like parables, don't read this book. Go read a book of Hasidic tales collected by Martin Buber, a book of Sufi stories collected by Idries Shah, or a book of parables and sayings by Anthony de Mello instead.

Or then again, you could just try Jesus. Jesus is always good.""", """ I need to start this review by stating 1) I can't stand self-help books and 2) I'm a feminist (no, I don't hate men- some men are quite awesome, but I am very conscious of women and our place in the world.)

Short summary (mild spoilers): A boy named Santiago follows his 'Personal Legend' in traveling from Spain to the Pyramids in Egypt searching for treasure. Along the way, he learns 'the Language of the World' the 'Soul of the World' and discovers that the 'Soul of God' is 'his own soul.'

If the statements in quotes above ('personal legend', etc) fascinate you, then you'll enjoy this book. If you think they are hokey and silly, then you'll think this is a terrible book. If you think statements such as "When you want something, all the universe conspires you to achieve it" and "All things are one" are moving and life-changing, you'll love this book. If such statements have you rolling your eyes, then this isn't your cup of tea.

Its not that I find anything wrong with these messages. They are important, but must be balanced with responsibility. In my experience, 'following your dreams' (or personal legend) is not the only way toward wisdom and strength. Is the person who struggles to put food on the table every day for his or her family, consciously realizing that he or she may not be following his or her 'personal legend' any less heroic than some traveler who leaves everything and everyone he or she is responsible for to go on a spiritual quest? Coelho comes close to labeling such people, as losers in life, which I find completely off the mark as some of these people have the most to offer in terms of wisdom.

The issue of responsibility is also part of this book's sexism. The main male characters in the novel have 'Personal Legends' - they are either seeking them, or have achieved them, or have failed to achieve them. But Coelho never mentions 'Personal Legend' with regard to women, other than to say that Fatima, Santiago's fiance, is 'a part of Santiago's Personal Legend." Thats fine, but what about her own Personal Legend? Instead of traveling to find her dreams, she is content to sit around, do chores, and stare everyday at the desert to wait for his return. This is her 'fate' as a desert women. The fact that women don't have Personal Legends is even more galling considering the fact that according to Coelho, even minerals such as lead and copper have Personal Legends, allowing them to 'evolve' to something better (ie, gold).

In the ideal world presented in THE ALCHEMIST, it seems that the job of men is to seek out their personal legends, leaving aside thoughts of family and responsibility, and its the job of women to let them, and pine for their return. Of course, someone has to do the unheroic, inconvenient work of taking care of the children, the animals, the elderly, the ill...If everyone simply goes off on spiritual quests, deciding they have no responsibility other than to seek their Personal Legends, no one would be taking responsibility for the unglamorous work that simply has to take place for the world to run.

On the other hand, what if both men and women are allowed to struggle towards their 'Personal Legends,' and help each other as best as they can towards them, but recognize that their responsibilities may force them to defer, compromise, or even 'sacrifice' their dreams? This may seem depressing, but it isn't necessarily. Coelho seems to think that Personal Legends are fixed at childhood (or at birth, or even before) and are not changeable: they have to be followed through to the end, no matter how silly. But in my experience, many people have chosen to adjust, compromise, and even 'give up' on their dreams, only to find that life grants them something better, or they have a new, better dream to follow, a path providing greater wisdom. For me, these people have a more realistic, more humble, more fair, and less cliched vision of the world than Paulo Coelho's vision in THE ALCHEMIST."""]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim = 0).item()

# Exemplu de utilizare
similarity = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])
>>>>>>> dbab91eacac44fef15f6b5bb7f712bd5254f5d10
print(f"Similaritatea Cosine între cele două propoziții: {similarity:.4f}")