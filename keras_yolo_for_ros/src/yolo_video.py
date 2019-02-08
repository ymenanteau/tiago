#!/usr/bin/env python

import sys
import argparse
import rospy
import io
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.msg as msg
from std_msgs.msg import String
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image

FLAGS = None
bridge = CvBridge()
image = None

def detect_img(data):
    global image
    if image != None:
	return
    img_cv = bridge.imgmsg_to_cv2(data, "bgr8")
    img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_cv)
    # r_image.show()
    # yolo.close_session()

def callback(data):
    detect_img(data)
    
def listener():
    global image
    yolo = YOLO(**vars(FLAGS))
    rospy.init_node('yolo', anonymous=True)
    pub = rospy.Publisher('yolo_res', String, queue_size=10)
    rospy.Subscriber("/camera/rgb/image_raw", msg.Image, callback)
    print("Waiting ...")
    while True:
	if image != None:
		res, rimage = yolo.detect_image(image)
		rimage.show()
	   	image = None
           	message = "{ "
	   	for obj in res:
			top, left, bottom, right, label, score = obj
			message = message + "{top : " + str(top) + ", left : " + str(left) + ", bottom : " + str(bottom)
			message = message + ", right : " + str(right) + ", label : " + str(label) + ", score : " + str(score) + "},"
		message = message[:-1]
		message = message + "}"
		rospy.loginfo(message)
   		pub.publish(message)

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    FLAGS = parser.parse_args()
    listener()
