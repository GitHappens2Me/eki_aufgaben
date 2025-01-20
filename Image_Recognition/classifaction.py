import torch 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch import nn, optim
from tqdm import tqdm

# Mit Faltungsnetzen

if(torch.cuda.is_available()): 
    device = "gpu" 
else:
    device ="cpu"

# Transformationsmethode angewendet auf die Daten bevor sie geladen werden
# Wir ändern die PIL-Images in Tensoren und ändern die Bytes in Floats
# Die Syntax nutzen wir, da man auch mehrere transformationen anwenden kann
trainings_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])

# Laden der Daten in ImageFolder (DataSet mit Extra Funktionen)
# Wir nutzen trainings_transform um Bilder zu Tensoren zu machen
dataset = ImageFolder(".\CIFAR10\CIFAR-10-images\\train", transform=trainings_transform)

dataset_test = ImageFolder(".\CIFAR10\CIFAR-10-images\\test", transform=trainings_transform)

# DataSets verhalten sich wie Arrays
# Ausgabe: (<PIL.Image.Image image mode=RGB size=32x32 at 0x1437F4BA250>, 0)   
#           Klasse: 0 
#print(dataset[0])

# Dataloader erstellen 
# Dataloader verhalten sich wie Iteratoren (mit extra Funktionen)
# Wir definieren dass immer _batch_size_ Bilder geladen werden (hier 32 oder 256)
trainings_set = DataLoader(dataset, batch_size = 256, shuffle = True)
#print(trainings_set)

test_set = DataLoader(dataset_test, batch_size = 256, shuffle = True)

# Dataloader gibt direkt 32 Bilder und die entsprechenden Labels zurück
for batch, label in trainings_set:
    #print(batch.shape)
    #print(label.shape)
    # Ausgabe:
    # torch.Size([32, 3, 32, 32])  ->    ([Batchsize, Farbkanäle, X, Y])
    # torch.Size([32])   -> 32 Label der Bilder
    break

class Net(nn.Module):

    # Hier definieren wir alle Layer 
    def __init__(self):
        super().__init__()
        # Flatten ändert die (3, 32, 32)- Matrix in einen 1d Vektor mit 3072 Zahlen
        self.flat = nn.Flatten()
        # Fully Connected Layer vom Network
        self.fc1 = nn.Linear(in_features=32*32*3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.relu = nn.ReLU()

    # Forward definiert was im Netzwerk passiert
    # 1) Flatten
    # 2) fully connected Layer (fc1)
    # 3) Relu (nicht lineare Aktivieren)
    # 4) fully connected Layer (fc2)
    def forward(self, x):
        x = self.relu(self.fc1(self.flat(x)))
        return self.fc2(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding="same")

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=64)


        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, img):
        # Faltung, Pooling und relu
        img = self.relu(self.bn1(self.pool(self.conv1(img)))) # 8x16x16
        img = self.relu(self.bn2(self.pool(self.conv2(img)))) # 16x8x8
        img = self.relu(self.bn3(self.pool(self.conv3(img)))) # 32x4x4
        img = self.relu(self.bn4(self.pool(self.conv4(img)))) # 64x2x2 = 256 dim
        
        img = self.fc2(self.relu(self.fc1(self.flat(img))))
        return img

# Netzwerk konkret instanzieren
# net = Net()
net = CNN()
# Optimizer werden alle Parameter des Netzwerks gegeben und eine Lernrate
# Pytorch trackt die Gradienten mit. Der Optimizer updated alle Parameter bei optim.step() (zieht den Gradienten ab). 
# Wir haben in der Vorlesung Stochastic Gradiant Descend (SGD). Adam-Optimizer macht noch extra schritte
optimizer = optim.Adam(params=net.parameters(), lr=0.01)

# Wir defininieren die Lossfunktion
criterion = nn.CrossEntropyLoss()



def epoch(epoch_index, modus, set):
    # Loss über die ganze Epoche
    total_loss = 0.0
    total_cnt = 0

    # Accuracy Tracker:
    total_correct = 0
    # Loading Bar
    bar = tqdm(set)

    # TrainingsLoop: 
    for batch, labels in bar:

        if(modus == "train"):
            # Gradienten auf 0 setzen
            optimizer.zero_grad()
        # Berechnet vorhersage für alle bilder im batch
        predictions = net(batch)
        
        
        # argmax gibt das neuron was am stärksten feuert
        # Die summe summiert alle True (1) auf
        total_correct += torch.sum(torch.argmax(predictions, dim=1) == labels)
        

        # Berechnet den Loss
        loss = criterion(predictions, labels)

        # Berechnet den Mittleren Loss über die ganze Epoche 
        total_loss += loss.item()
        total_cnt += 256
        # Accuracy updaten
        accuracy = total_correct / total_cnt
        bar.set_description(f"[{modus:>5}] epoch={epoch_index}, loss={1000*total_loss/total_cnt:.4f}, acc={100.0*accuracy:.2f}%") 
        #print(total_loss / total_cnt)

        if(modus == "train"):
            # Berechnet die Gradienten 
            loss.backward()
            # Updated die Parameter
            optimizer.step()
            #print(prediction.shape)


# EpochenLopo
for epoch_index in range(100):
    epoch(epoch_index, "train", trainings_set)
    epoch(epoch_index, "test", test_set)
    
    

