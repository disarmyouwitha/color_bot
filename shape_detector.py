import cv2

class shape_detector:
	def __init__(self):
		pass

	def detect(self, c):
		# [Initialize the shape name and approximate the contour]:
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		#print('_APPROX: {0}'.format(len(approx)))

		# [If the shape is a triangle, it will have 3 vertices]:
		if len(approx) == 3:
			shape = "triangle"

		# [If the shape has 4 vertices, it is either a square or a rectangle]:
		elif len(approx) == 4:
			# [Compute the bounding box of the contour and use the bounding box to compute the aspect ratio]:
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# [A square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle]:
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# [If the shape is a pentagon, it will have 5 vertices]:
		#elif len(approx) == 5:
		#	shape = "pentagon"

		# [otherwise, we assume the shape is a circle]:
		else:
			shape = "circle"

		# [Return the name of the shape]:
		return shape