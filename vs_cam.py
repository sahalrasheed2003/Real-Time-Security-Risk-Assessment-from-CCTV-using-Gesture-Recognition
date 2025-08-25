import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from myapp.DBConnection import Db
from myapp.aa import predictcnn

from PIL import Image
# output=["Man with gun", "safe", "Suspicious men", "safe", "Fighting men", "Emergency help"]
from myapp.train_weapon import predict_cnn_weapon

output=["No fight", "Fight"]
proj_path=r"D:\project\Risk_assessment\myapp\\"

# Importing Libraries
import cv2
# Importing Libraries
import cv2


cam_id="1"

app_email="riskassess35@gmail.com"
# mail_password="Risk@2k25"
app_password="xfyv xjkh hute kxwx"

import smtplib

def send_mail(email, msgg):
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(app_email, app_password)
    msg = MIMEMultipart()  # create a message.........."
    msg['From'] =app_email
    msg['To'] = email
    msg['Subject'] = "Risk alert"
    body = msgg
    msg.attach(MIMEText(body, 'plain'))
    s.send_message(msgg)
    print("Okkk")
    return "ok"

listimg=[]

def startcam():
    countval = 0
    # Start capturing video from webcam
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    cap.set(3,800)
    cap.set(4,600)
    jj=0
    last_msg=""
    weap_cnt=0
    while True:
        # Read video frame by frame
        _, frame = cap.read()
        try:
            img1=frame.copy()

            xx1 = int(0.5 * frame.shape[1])
            xy1 = 10
            xx2 = frame.shape[1] - 10
            xy2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (xx1 - 1, xy1 - 1), (xx2 + 1, xy2 + 1), (255, 0, 0), 1)
            cv2image = frame # cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)


            # Flip image
            frame = cv2.flip(frame, 1)
            # Convert BGR image to RGB image
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # if hands are present in image(frame)
            # x1, y1, x2, y2 = 10000, 10000, 0, 0
            # x1, y1, x2, y2 = xx1, xy1, xx2, xy2
            jj = jj + 1
            # If landmarks list is not empty
            cv2image = cv2image[xy1: xy2, xx1: xx2]
            cv2.imwrite(proj_path + "sampleeee.jpg", cv2image)

            #       fight detection
            try:
                # print(jj)
                if jj >= 10:
                    # cv2.imwrite(proj_path + "sample.jpg",img1)
                    # # print("img1++++++++++===========================")
                    # im = Image.open(proj_path + "sample.jpg")
                    # im1 = im.crop( (x1,y1,x2, y2))
                    #
                    # # Shows the image in image viewer
                    # im1 = im1.save(proj_path + "samplecrop1.jpg")
                    #
                    jj=0
                    image = cv2.imread(proj_path + "sampleeee.jpg")
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    invert = cv2.bitwise_not(gray)  # OR
                    # invert = 255 - image


                    # Setting parameter values
                    t_lower = 50  # Lower Threshold
                    t_upper = 150  # Upper threshold

                    # Applying the Canny Edge filter
                    edge = cv2.Canny(invert, t_lower, t_upper)

                    invert = cv2.bitwise_not(edge)  # OR

                    cv2.imwrite(proj_path + "samplecrop2.jpg",invert)
                    # print("CC ", countval)
                    if countval<len(listimg):
                        res=predictcnn(listimg[countval])
                        countval=countval+1
                    else:
                        res = predictcnn(proj_path + "samplecrop2.jpg")
                        # res = predictcnn(r"C:\Users\athul\PycharmProjects\sign language recognition - Copy\dataSet\trainingData\Q\1.jpg")
                    # print(output[res[0]])
                    out_put=output[res]
                    print("Ress  ", out_put)

                    if out_put == "Fight":
                        # db code
                        d = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                        save_path = r"D:\project\Risk_assessment\myapp\static\detections\\" + d
                        pth = "/static/detections/" + d
                        cv2.imwrite(save_path, frame)
                        import vlc
                        p = vlc.MediaPlayer(r"D:\project\Risk_assessment\myapp\static\alarm-sound-effect.mp3")
                        p.play()
                        admin_mail="sanidhyasharlin@gmail.com"
                        db=Db()
                        ressss=db.selectOne("SELECT email FROM `myapp_authority`, `myapp_allocation`, `myapp_camera` WHERE `myapp_allocation`.AUTHORITY_id=`myapp_authority`.id AND `myapp_allocation`.PLACE_id=`myapp_camera`.PLACE_id AND `myapp_camera`.id='"+cam_id+"'")
                        authority_mail=ressss['email']
                        msg="Fighting detected in Camera " + cam_id

                        print(last_msg,"opo")

                        db = Db()
                        db.insert(
                            "insert into myapp_detections values(null, curdate(), curtime(), '" + pth + "','" + str(
                                cam_id) + "','Detected')")

                        if out_put !=last_msg:
                            last_msg = out_put
                            print(admin_mail, msg)
                            print(authority_mail, msg)
                            # send_mail(admin_mail, msg)
                            # send_mail(authority_mail, msg)
                            # db = Db()
                            # db.insert(
                            #     "insert into myapp_detections values(null, curdate(), curtime(), '" + pth + "','" + str(
                            #         cam_id) + "','" + out_put + "')")

                    # weapon detection
                    res2, scr = predict_cnn_weapon(proj_path + "samplecrop2.jpg")
                    wpn=["No", "Yes"]
                    print("Pred weapon  ", res, wpn[res2])
                    if scr > 0.90 and res2 !=0:
                        weap_cnt+=1
                        print("wc  ", weap_cnt)
                        if weap_cnt >=5:


                            print(res2, scr)
                            wpn=["No", "Yes"]
                            admin_mail = "sanidhyasharlin@gmail.com"
                            db = Db()
                            res = db.selectOne(
                                "SELECT email FROM `myapp_authority`, `myapp_allocation`, `myapp_camera` WHERE `myapp_allocation`.AUTHORITY_id=`myapp_authority`.id AND `myapp_allocation`.PLACE_id=`myapp_camera`.PLACE_id AND `myapp_camera`.id='" + cam_id + "'")
                            authority_mail = res['email']
                            msg = "Weapon detected in Camera " + cam_id
                            d = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                            save_path = r"D:\project\Risk_assessment\myapp\static\detections\\" + d
                            pth = "/static/detections/" + d
                            cv2.imwrite(save_path, frame)
                            print(admin_mail, msg)
                            print(authority_mail, msg)
                            # send_mail(admin_mail, msg)
                            # send_mail(authority_mail, msg)
                            db = Db()
                            db.insert(
                                "insert into myapp_detections values(null, curdate(), curtime(), '" + pth + "','" + str(
                                    cam_id) + "','Detected')")
                            weap_cnt=0

                    else:
                        weap_cnt=0



            except Exception as e:
                print(e)

                # set brightness
                # sbc.set_brightness(int(b_level))

            # Display Video and when 'q' is entered,
            # destroy the window
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (350, 100)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2
            # print("=======",txt)
            # Using cv2.putText() method
            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                ####            main code
                break
        except Exception as e:
            print(e)
            break


startcam()
