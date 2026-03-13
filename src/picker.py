import cv2
import json

parking_slots = []
current_slot = []

def mouse_click(event, x, y, flags, param):
    global parking_slots, current_slot

    if event == cv2.EVENT_LBUTTONDOWN:
        current_slot.append((x, y))
        print(f"Punt toegevoegd: ({x}, {y})")

        if len(current_slot) == 4:
            parking_slots.append(current_slot.copy())
            print(f"Punt toegevoegd aan parkeerplaats: {current_slot}")
            current_slot.clear()

    if event == cv2.EVENT_RBUTTONDOWN:
        if current_slot:
            removed_point = current_slot.pop()
            print(f"Punt verwijderd: {removed_point}")
        elif parking_slots:
            removed_slot = parking_slots.pop()
            print(f"Parkeerplaats verwijderd: {removed_slot}")

img_path = 'data/raw/nearly_empty/test.jpg'

cv2.namedWindow('Parking Slot Picker: links=punt, rechts=undo, s=opslaan')
cv2.setMouseCallback('Parking Slot Picker: links=punt, rechts=undo, s=opslaan', mouse_click)

while True:
    img = cv2.imread(img_path)

    for slot in parking_slots:
        for i in range(4):
            pt1 = slot[i]
            pt2 = slot[(i + 1) % 4]
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    for pos in current_slot:
        cv2.circle(img, pos, 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow('Parking Slot Picker: links=punt, rechts=undo, s=opslaan', img)

    key = cv2.waitKey(1)

    if key == ord('s'):
        with open('config/test/parking_slots.json', 'w') as f:
            json.dump(parking_slots, f, indent=4)
        print("Parkeerplaatsen opgeslagen")
        break

    elif key == 27:  # ESC toets om te stoppen
        print("Afgebroken door gebruiker.")
        break

cv2.destroyAllWindows()