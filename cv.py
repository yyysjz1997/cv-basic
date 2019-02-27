import numpy as np
from matplotlib.pyplot import plot, show

x = np.linspace(0, 2 * np.pi, 30)#创建一个包含30个点的余弦波信号
wave = np.cos(x)
transformed = np.fft.fft(wave)#使用fft函数对余弦波信号进行傅里叶变换。
print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))#对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
plot(transformed)#使用Matplotlib绘制变换后的信号。
show()


'''
import cv2
import os

path = "D:\Pattern_recognition\MTCNN-Tensorflow\data\htest\lfpw_testImage"
gt_imdb = []

def gogo():
	for item in os.listdir(path):
		gt_imdb.append(os.path.join(path, item))
	for imagepath in gt_imdb:
		print(imagepath)
		img = cv2.imread(imagepath)
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
		eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
		#img = cv2.imread('yyy.jpg')
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
	gogo()
'''

'''
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
				help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
				help="path to input video file")
ap.add_argument("-t", "--threshold", type=float, default=0.27,
				help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type=int, default=2,
				help="the number of consecutive frames the eye must be below the threshold")


def main():
	args = vars(ap.parse_args())
	EYE_AR_THRESH = args['threshold']
	EYE_AR_CONSEC_FRAMES = args['frames']

	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	print("[INFO] print q to quit...")
	if args['video'] == "camera":
		vs = VideoStream(src=0).start()
		fileStream = False
	else:
		vs = FileVideoStream(args["video"]).start()
		fileStream = True

	time.sleep(1.0)

	# loop over frames from the video stream
	while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		if fileStream and not vs.more():
			break

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]

			# average the eye aspect ratio together for both eyes

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			#cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
						#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()


if __name__ == '__main__':
	main()
'''


'''
# import the necessary packages
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
				help="path to facial landmark predictor")

path = "D:\Pattern_recognition\MTCNN-Tensorflow\data\htest\lfpw_testImage"
gt_imdb = []




def main():
	args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	print("[INFO] print q to quit...")

	time.sleep(1.0)



	# loop over frames from the video stream
	while True:
		for item in os.listdir(path):
			gt_imdb.append(os.path.join(path, item))
		for imagepath in gt_imdb:
			print(imagepath)
			frame = cv2.imread(imagepath)
			# grab the frame from the threaded video file stream, resize
			# it, and convert it to grayscale
			# channels)
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# detect faces in the grayscale frame
			rects = detector(gray, 0)

			# loop over the face detections
			for rect in rects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]

				# average the eye aspect ratio together for both eyes

				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


			# show the frame
			cv2.imshow("img", frame)

			cv2.waitKey(0)
			cv2.destroyAllWindows()

			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	# do a bit of cleanup
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
'''