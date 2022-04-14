import cv2
import face_recognition
import numpy as np
import os
import datetime

from main import sheet, book

video_capture = cv2.VideoCapture(0)

image_1 = face_recognition.load_image_file("Image/a2.png")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]

image_2 = face_recognition.load_image_file("Image/b2.png")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]

image_3 = face_recognition.load_image_file("Image/p2.png")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

known_face_encodings = [

    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding,
]
known_face_names = [

    "Abiy AHMED",
    "Barack Obama",
    "Vladmir Puttin",
]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

now = datetime.datetime.now()
today = now.day
month = now.month


def markattendance(name):
    with open('attendance.csv', 'r+', ) as f:
        mydatalist = f.readline()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
            if name not in namelist:
                now = datetime.now()
                dtstring = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name}m{dtstring}')


while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            # Assign attendance
            if name in range(1, 61):
                sheet.cell(row=name, column=int(today)).value = "attendance"
            else:
                pass

        face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        markattendance(name)
    cv2.imshow('Video', frame)

    book.save(str(month) + '.csv')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
