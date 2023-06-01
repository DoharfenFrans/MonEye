import cv2
import numpy as np
from playsound import playsound
from flask import Flask, render_template, Response, request
from camera2 import VideoCamera
from gtts import gTTS
import os
import json

MIN_POINT = 35

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

        
def gen(camera):
    img_source, temp_kp_desc, video = camera.get_frame()
    img_final = None

    def feature():
        # detector = cv2.ORB_create() 
        # FLANN_INDEX_KDTREE = 1
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # detector = cv2.ORB_create() 
        detector = cv2.xfeatures2d.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L1) 
        return detector, matcher

    def filter_matches(kp1, kp2, matches, ratio=0.75):  
        mkp1, mkp2 = [], []
        for m,n in matches:
            if m.distance < n.distance * ratio:
                good_point = m #jika m=m[1] maka bentuk kotak hijaunya tidak karuan
                mkp1.append(kp1[good_point.queryIdx])
                mkp2.append(kp2[good_point.trainIdx])
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs

    def explore_match(img1, img2, kp_pairs, status=None, H=None, showText=None):
        h1, w1 = img1.shape #150x300
        h2, w2 = img2.shape #480x640
                
        if H is not None:
            corner = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corner = np.int32(cv2.perspectiveTransform(corner.reshape(1, -1, 2), H).reshape(-1, 2))
            cv2.polylines(img_final, [corner], True, (0, 255, 0), 2)
            cv2.fillPoly(img2, [corner], (255, 255, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_final, showText, (corner[0][0], corner[0][1]), font, 1, (0, 255, 0), 2)

    def text():
        cv2.putText(img_final, 'TOTAL : '+str(total), (10,470), cv2.FONT_HERSHEY_TRIPLEX, 1, (2, 62, 28), 4)
        cv2.putText(img_final, 'Arahkan uang ke depan kamera', (150,20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (2, 62, 28), 4)
        cv2.putText(img_final, 'TOTAL : '+str(total), (10,470), cv2.FONT_HERSHEY_TRIPLEX, 1, (10, 148, 52), 1)
        cv2.putText(img_final, 'Arahkan uang ke depan kamera', (150,20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (10, 148, 52), 1)
        # cv2.putText(img4, 'ARAHKAN KAMERA KE UANG AGAR NOMICAL DAPAT TERBACA', (25,75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        
    def match_and_draw(checksound, found, count):
        kp2, desc2 = detector.detectAndCompute(img2, None)
        searchIndex = 0
        total = 0

        while(True):
            if not found :
                searchIndex += 1
            if searchIndex > 28:
                break
            if found:
                kp2, desc2 = detector.detectAndCompute(img2, None)

            kp1 = temp_kp_desc[searchIndex-1][0]
            desc1 = temp_kp_desc[searchIndex-1][1]
            img1 = img_source[searchIndex-1]
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

            if searchIndex > 0 and searchIndex < 5:
                showText = 1000               
            elif searchIndex <= 8:
                showText = 2000
            elif searchIndex <= 12:
                showText = 5000
            elif searchIndex <= 16:
                showText = 10000
            elif searchIndex <= 20:
                showText = 20000
            elif searchIndex <= 24:
                showText = 50000
            elif searchIndex <= 28:
                showText = 100000
            
            if len(p1) >= MIN_POINT:
                #match ketemu
                if not found:
                    checksound = True
                found = True
                count = count + 1
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                explore_match(img1, img2, kp_pairs, status, H, "Rp"+str(showText))
                total += showText
                showText = 0
            else:
                #belum ketemu match
                found = False
                checksound = False
                H, status = None, None
                count = 0

        return found, checksound, count, total

    # Buat cek perbedaan banyak keypoint kalo ganti detector
    def draw_cross_keypoints(img, keypoints, color):
        """ Draw keypoints as crosses, and return the new image with the crosses. """
        img_kp = img.copy()  # Create a copy of img

        # Iterate over all keypoints and draw a cross on evey point.
        for kp in keypoints:
            x, y = kp.pt  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

            x = int(round(x))  # Round an cast to int
            y = int(round(y))

            # Draw a cross with (x, y) center
            cv2.drawMarker(img_kp, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)

        return img_kp  # Return the image with the drawn crosses.

    detector, matcher = feature()
    checksound = True
    found = False
    searchIndex = 1
    count = 0
    finalfound = False

    while (True): 

        ret, frame = video.read()
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_final = frame

        # Buat cek perbedaan banyak keypoint kalo ganti detector
        # kp2, desc2 = detector.detectAndCompute(img2, None)
        # img_final = draw_cross_keypoints(img_final, kp2, color=(120,157,187))  # Draw keypoints as "+" signs
        # img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)

        found, checksound, count, total = match_and_draw(checksound, found, count)
        text()

        #untuk convert text ke suara
        with open("text.txt", "w") as f:
            print(total)
            if total == 0:
                f.write("tidak ada uang terdeteksi")
            else :
                f.write(str(total)+" rupiah")


        fh = open("text.txt", "r")
        myText = fh.read().replace("\n", " ")
        language = 'id'
        output = gTTS(text=myText, lang=language, slow=False)
        # output.save("output.mp3")
        output.save("%s.mp3" % os.path.join("static","output"))
        fh.close()

        #untuk mengirim frame gambar ke html
        ret, jpeg = cv2.imencode('.jpg', img_final)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() 
            + b'\r\n\r\n')


# @app.route('/processTrigger/<string:alphabet>', methods=['POST'])
# def processTrigger(alphabet):
#     alphabet = json.loads(alphabet)
#     os.system("start output.mp3")
#     return 'SUCCESS'    
            

@app.route('/video_feed')
def video_feed():
    print("*")
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame') #kasi tau tipe respon apa yg bakal diterima html

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port='5000', debug=True)