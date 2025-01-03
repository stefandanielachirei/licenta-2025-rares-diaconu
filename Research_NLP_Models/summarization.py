<<<<<<< HEAD
import subprocess
import os
import warnings
import torch
import tensorflow as tf
from transformers import pipeline

warnings.filterwarnings("ignore")

# Configurarea GPU pentru TensorFlow
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

# Inițializăm pipeline-ul de summarization pe GPU-ul specificat
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device.index)

# Textul pe care vrem să-l rezumăm
Review_Summary = """ I need to start this review by stating 1) I can't stand self-help books and 2) I'm a feminist (no, I don't hate men- some men are quite awesome, but I am very conscious of women and our place in the world.)

Short summary (mild spoilers): A boy named Santiago follows his 'Personal Legend' in traveling from Spain to the Pyramids in Egypt searching for treasure. Along the way, he learns 'the Language of the World' the 'Soul of the World' and discovers that the 'Soul of God' is 'his own soul.'

If the statements in quotes above ('personal legend', etc) fascinate you, then you'll enjoy this book. If you think they are hokey and silly, then you'll think this is a terrible book. If you think statements such as "When you want something, all the universe conspires you to achieve it" and "All things are one" are moving and life-changing, you'll love this book. If such statements have you rolling your eyes, then this isn't your cup of tea.

Its not that I find anything wrong with these messages. They are important, but must be balanced with responsibility. In my experience, 'following your dreams' (or personal legend) is not the only way toward wisdom and strength. Is the person who struggles to put food on the table every day for his or her family, consciously realizing that he or she may not be following his or her 'personal legend' any less heroic than some traveler who leaves everything and everyone he or she is responsible for to go on a spiritual quest? Coelho comes close to labeling such people, as losers in life, which I find completely off the mark as some of these people have the most to offer in terms of wisdom.

The issue of responsibility is also part of this book's sexism. The main male characters in the novel have 'Personal Legends' - they are either seeking them, or have achieved them, or have failed to achieve them. But Coelho never mentions 'Personal Legend' with regard to women, other than to say that Fatima, Santiago's fiance, is 'a part of Santiago's Personal Legend." Thats fine, but what about her own Personal Legend? Instead of traveling to find her dreams, she is content to sit around, do chores, and stare everyday at the desert to wait for his return. This is her 'fate' as a desert women. The fact that women don't have Personal Legends is even more galling considering the fact that according to Coelho, even minerals such as lead and copper have Personal Legends, allowing them to 'evolve' to something better (ie, gold).

In the ideal world presented in THE ALCHEMIST, it seems that the job of men is to seek out their personal legends, leaving aside thoughts of family and responsibility, and its the job of women to let them, and pine for their return. Of course, someone has to do the unheroic, inconvenient work of taking care of the children, the animals, the elderly, the ill...If everyone simply goes off on spiritual quests, deciding they have no responsibility other than to seek their Personal Legends, no one would be taking responsibility for the unglamorous work that simply has to take place for the world to run.

On the other hand, what if both men and women are allowed to struggle towards their 'Personal Legends,' and help each other as best as they can towards them, but recognize that their responsibilities may force them to defer, compromise, or even 'sacrifice' their dreams? This may seem depressing, but it isn't necessarily. Coelho seems to think that Personal Legends are fixed at childhood (or at birth, or even before) and are not changeable: they have to be followed through to the end, no matter how silly. But in my experience, many people have chosen to adjust, compromise, and even 'give up' on their dreams, only to find that life grants them something better, or they have a new, better dream to follow, a path providing greater wisdom. For me, these people have a more realistic, more humble, more fair, and less cliched vision of the world than Paulo Coelho's vision in THE ALCHEMIST. """

# Rulăm sumarizarea
print(summarizer(Review_Summary, max_length=130, min_length=30, do_sample=False))
=======
import subprocess
import os
import warnings
import torch
import tensorflow as tf
from transformers import pipeline

warnings.filterwarnings("ignore")

# Configurarea GPU pentru TensorFlow
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

# Inițializăm pipeline-ul de summarization pe GPU-ul specificat
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device.index)

# Textul pe care vrem să-l rezumăm
Review_Summary = """ I need to start this review by stating 1) I can't stand self-help books and 2) I'm a feminist (no, I don't hate men- some men are quite awesome, but I am very conscious of women and our place in the world.)

Short summary (mild spoilers): A boy named Santiago follows his 'Personal Legend' in traveling from Spain to the Pyramids in Egypt searching for treasure. Along the way, he learns 'the Language of the World' the 'Soul of the World' and discovers that the 'Soul of God' is 'his own soul.'

If the statements in quotes above ('personal legend', etc) fascinate you, then you'll enjoy this book. If you think they are hokey and silly, then you'll think this is a terrible book. If you think statements such as "When you want something, all the universe conspires you to achieve it" and "All things are one" are moving and life-changing, you'll love this book. If such statements have you rolling your eyes, then this isn't your cup of tea.

Its not that I find anything wrong with these messages. They are important, but must be balanced with responsibility. In my experience, 'following your dreams' (or personal legend) is not the only way toward wisdom and strength. Is the person who struggles to put food on the table every day for his or her family, consciously realizing that he or she may not be following his or her 'personal legend' any less heroic than some traveler who leaves everything and everyone he or she is responsible for to go on a spiritual quest? Coelho comes close to labeling such people, as losers in life, which I find completely off the mark as some of these people have the most to offer in terms of wisdom.

The issue of responsibility is also part of this book's sexism. The main male characters in the novel have 'Personal Legends' - they are either seeking them, or have achieved them, or have failed to achieve them. But Coelho never mentions 'Personal Legend' with regard to women, other than to say that Fatima, Santiago's fiance, is 'a part of Santiago's Personal Legend." Thats fine, but what about her own Personal Legend? Instead of traveling to find her dreams, she is content to sit around, do chores, and stare everyday at the desert to wait for his return. This is her 'fate' as a desert women. The fact that women don't have Personal Legends is even more galling considering the fact that according to Coelho, even minerals such as lead and copper have Personal Legends, allowing them to 'evolve' to something better (ie, gold).

In the ideal world presented in THE ALCHEMIST, it seems that the job of men is to seek out their personal legends, leaving aside thoughts of family and responsibility, and its the job of women to let them, and pine for their return. Of course, someone has to do the unheroic, inconvenient work of taking care of the children, the animals, the elderly, the ill...If everyone simply goes off on spiritual quests, deciding they have no responsibility other than to seek their Personal Legends, no one would be taking responsibility for the unglamorous work that simply has to take place for the world to run.

On the other hand, what if both men and women are allowed to struggle towards their 'Personal Legends,' and help each other as best as they can towards them, but recognize that their responsibilities may force them to defer, compromise, or even 'sacrifice' their dreams? This may seem depressing, but it isn't necessarily. Coelho seems to think that Personal Legends are fixed at childhood (or at birth, or even before) and are not changeable: they have to be followed through to the end, no matter how silly. But in my experience, many people have chosen to adjust, compromise, and even 'give up' on their dreams, only to find that life grants them something better, or they have a new, better dream to follow, a path providing greater wisdom. For me, these people have a more realistic, more humble, more fair, and less cliched vision of the world than Paulo Coelho's vision in THE ALCHEMIST. """

# Rulăm sumarizarea
print(summarizer(Review_Summary, max_length=130, min_length=30, do_sample=False))
>>>>>>> dbab91eacac44fef15f6b5bb7f712bd5254f5d10
