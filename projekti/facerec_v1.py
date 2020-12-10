import face_recognition
import cv2
import numpy as np

# 0==default viittaus
video_capture = cv2.VideoCapture(0)

# Kuvahommii
mikko_image = face_recognition.load_image_file("mikko.jpg")
mikko_face_encoding = face_recognition.face_encodings(mikko_image)[0]

# Arrayt naamojen enkoodaukselle 
known_face_encodings = [
    mikko_face_encoding,
]
known_face_names = [
    "Mikko Korpi"
]

# muuttujien alustus
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # yhden framen vienti
    ret, frame = video_capture.read()

    # pilkotaan 1/4 osaan, saadaan nopeampi kuva
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BGR-RGB translaatio, opencvsta facerecciin
    rgb_small_frame = small_frame[:, :, ::-1]

    # prosessoidaan vain joka toinen freimi
    if process_this_frame:
        # etsii kaikki naamat ja enkoodaukset videosta
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # matchausta
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Tuntematon"

            # # jos matchaus onnistui ekassa enkoodauksessa, kaytetaan sita.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            #kaytetaan tunnistettua naamaa matkan perusteella
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # tulos
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Skeilataan takas ylos
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # nelionpiirtaja
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Annetaan etiketti
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # naytetaan tulos
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
