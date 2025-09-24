from Src.Preprocessing import Preprocessing
from Src.Dataset import FakeNewsDataset
from Src.Model import load_model
from Src.Inference import predict_fake_news
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your preprocessing class
from Src.Preprocessing import preprocessing

# Load data
df = preprocessing.data_read()
texts = df['contents'].tolist()
labels = df['label'].tolist()

# Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

#load model and tokenizer
model, tokenizer, device = load_model("best_bert_model.pkl")

#tokenize the dataset
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)
test_dataset = FakeNewsDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=8)



