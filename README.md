# Real-time Object Tracking & Counting using YOLO11

Complete project for **Real-time Object Detection + Tracking + Counting**.

---

## Features

- Real-time webcam tracking
- Unique ID tracking (ByteTrack)
- Live object counting (current frame + unique total)
- Trajectory visualization
- Image based counting
- Easy model switching (yolo11n / yolo11s / yolo11m)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Real-time Tracking (Webcam)

```bash
python main.py
```

- Press `q` → Quit  
- Press `s` → Save screenshot

### 2. Count objects in an Image

```bash
python count_from_image.py your_image.jpg
```

### 3. Use Video File

`main.py` mein change karo:

```python
SOURCE = "videos/your_video.mp4"
SAVE_OUTPUT = True
```

---

## Model Options

| Model        | Speed     | Accuracy   | Recommended For      |
|--------------|-----------|------------|----------------------|
| yolo11n.pt   | Fastest   | Good       | Low end devices      |
| **yolo11s.pt** | Fast    | Better     | **Best for most users** |
| yolo11m.pt   | Medium    | High       | High accuracy need   |

`main.py` mein `MODEL_PATH` change karke model switch kar sakte ho.

---

## Project Structure

```
YOLO_Object_Counting_Project_Final/
├── main.py                 ← Main real-time tracking
├── count_from_image.py     ← Image counting
├── requirements.txt
├── README.md
├── models/                 ← Custom models yahan rakho
├── videos/                 ← Input videos
└── output/                 ← Results save hote hain
```

---

## Detectable Objects (COCO classes)

Person, Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck, Boat,  
Traffic light, Fire hydrant, Stop sign, Parking meter, Bench, Bird, Cat, Dog,  
Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe, Backpack, Umbrella,  
Handbag, Tie, Suitcase, Frisbee, Skis, Snowboard, Sports ball, Kite,  
Baseball bat, Baseball glove, Skateboard, Surfboard, Tennis racket, Bottle,  
Wine glass, Cup, Fork, Knife, Spoon, Bowl, Banana, Apple, Sandwich, Orange,  
Broccoli, Carrot, Hot dog, Pizza, Donut, Cake, Chair, Couch, Potted plant,  
Bed, Dining table, Toilet, TV, Laptop, Mouse, Remote, Keyboard, Cell phone,  
Microwave, Oven, Toaster, Sink, Refrigerator, Book, Clock, Vase, Scissors,  
Teddy bear, Hair drier, Toothbrush

---

## Tips

- Better accuracy ke liye `yolo11s.pt` ya `yolo11m.pt` use karo
- Confidence kam karne ke liye `CONFIDENCE = 0.35` set karo
- Specific classes ke liye `TARGET_CLASSES = [0, 73, 39]` (person, book, bottle)

---

**Project Complete**  
Enjoy tracking!
