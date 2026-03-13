# Smart Parking Space Detector

Dit project maakt gebruik van Computer Vision en Deep Learning (PyTorch) voor de automatische detectie van parkeerplaatsbezetting op basis van camerabeelden.

## Projectoverzicht
Het systeem is ontworpen om parkeervakken te identificeren en per vak te bepalen of deze vrij of bezet is. Dit wordt bereikt door een combinatie van handmatige annotatie van de vakken en een neuraal netwerk dat getraind is op visuele kenmerken van voertuigen.

## Functionaliteiten
* **Picker Tool**: Een hulpmiddel voor het handmatig definiëren van parkeervakken via een grafische interface.
* **Classificatie**: Gebruik van een ResNet18-architectuur voor onderscheid tussen de klassen 'space-empty' en 'space-occupied'.
* **Inference**: Real-time analyse van uitsnedes (crops) met weergave van de status en de betrouwbaarheidsscore (confidence score).
* **Geoptimaliseerde Interface**: Tekstuele weergave die versprongen wordt getoond om overlapping bij dichtbijgelegen vakken te voorkomen.

## Installatie

### 1. Vereisten
Zorg dat Python 3.x geïnstalleerd is op het systeem.

### 2. Afhankelijkheden installeren
Installeer de noodzakelijke bibliotheken met het volgende commando:
```bash
pip install torch torchvision opencv-python numpy
```
## Gebruiksaanwijzing

### 1. Vakken definieren (picker.py)
Met de picker-tool worden de coördinaten van de parkeervakken vastgelegd in een configuratiebestand.
* **Linkermuisknop**: Voeg een punt toe aan het huidige vak.
* **Rechtermuisknop**: Verwijder het laatst toegevoegde punt of het laatst toegevoegde vak.
* **S-toets**: Sla de gedefinieerde vakken op in `config/parking_slots.json`.
### 2. Model trainen (train_torch.py)
Het model is getraind middels Transfer Learning op een dataset van parkeerbeelden.
* **inputresolutie**: 64x64 pixels.
* **optimizer**: Adam met een learning rate van 0.001.
* **output**: Het getrainde model wordt opgeslagen als `models/parking_model.pth`.
### 3. Detectie uitvoeren (detector.py)
Het hoofdprogramma laadt het getrainde model en de opgeslagen coördinaten om de analyse uit te voeren op nieuwe beelden of videoframes.
* De resultaten worden visueel weegegeven met groene (vrij) of rode (bezet) kaders.
* Per vak wordt de status (V/B) en de zekerheid van het model getoond.

##Projectstructuur
* src/
  * picker.py - Tool voor het definiëren van parkeervakken.
  * train_torch.py - Script voor het trainen van het model.
  * detector.py - Script voor het uitvoeren van de detectie.
* config/
  * parking_slots.json - Configuratiebestand met de coördinaten van de parkeervakken.
* models/
  * parking_model.pth - Het getrainde modelbestand.
