import numpy as np
import queue   


# def find_start_point(img):
#   calculate_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
#   matching_color = np.array(([ 61,  95, 154]))
#   y, x = img.shape[:-1]
#   x = x//2
#   for i in range(y):
#     if calculate_distance(img[i, x], matching_color) < 10:
#       return (i, x)


def bfs(img, start):
  calculate_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
  dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
  dst = np.zeros(img.shape[:-1])
  q = queue.Queue()
  q.put(start)
  
  while not q.empty():
    current = q.get()
    for dir in dirs:
      next_pos = (current[0] + dir[0], current[1] + dir[1])
      if 0 <= next_pos[0] < img.shape[0] and 0 <= next_pos[1] < img.shape[1]:
        if dst[next_pos] == 0 and calculate_distance(img[next_pos], img[current]) < 9:
          dst[next_pos] = 1
          q.put(next_pos)

  return dst  


def get_edge(img):
  height, width = 160, 120
  edgesL, edgesR = [], []
  for i in range(height):
    for j in range(width - 1):
      if img[i, j] == 0 and img[i, j+1] == 1:
        edgesL.append([j, i])
        break
  
  # for i in range(height):
  #   for j in range(width - 1, 0, -1):
  #     if img[i, j] == 0 and img[i, j-1] == 1:
  #       edgesR.append([j, i])
  #       break
      
  return np.array(edgesL) #, np.array(edgesR)  

import cv2
import time

cap = cv2.VideoCapture(0)

ang_filtered = 0
ang_filtered1 = 1
alpha = 0.6

while(True):
  ret, frame = cap.read()
  frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
  frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
  
  dst = bfs(frame, (frame.shape[0]//2-1, frame.shape[1]//2-1))
  
  edgesL = get_edge(dst)
  
  if len(edgesL) > 0:
    cv2.drawContours(frame, [edgesL], -1, (0, 0, 255), 2)    
  if edgesL.shape[0] > 5:
    x, y = edgesL[:,0], edgesL[:,1]
    A = np.vstack([y, np.ones(len(y))]).T
    m, c = np.linalg.lstsq(A, x, rcond=None)[0]
    print(m)

  # Show image
  cv2.imshow("Contours", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  time.sleep(0.1)

cap.release()

cv2.destroyAllWindows()

  
  # if contours and len(contours[0]) > 5:
  #   # Fit ellipse to contour
  #   ellipse = cv2.fitEllipse(contours[0])

  #   # Get angle of the fitted ellipse
  #   angle = ellipse[2]

  #   # print("Neck angle:", angle)
    
  #   if angle > 90:
  #     angle = 180 - angle 
    
  #   if np.abs(angle - ang_filtered) > 30:
  #     print('Skipped')
  #   else:
  #     ang_filtered = ang_filtered * alpha + angle * (1 - alpha)
  #     print("Neck angle:", ang_filtered)  
      
  # print(len(contours1))

  # if len(contours1) > 5:
  #   # Calculate the covariance matrix
  #   covariance, _, _ = np.cov(contours1[:,0,0], contours1[:,0,1], rowvar=False)

  #   # Perform eigendecomposition
  #   eigenvalues, eigenvectors = np.linalg.eigh(covariance)

  #   # Get the angle of the eigenvector with the largest eigenvalue
  #   angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

  #   # Make sure angle is within range (-90, 0]
  #   if angle < -45:
  #       angle += 90

  #   # Print angle
  #   print("Angle:", angle)
  
  
  # if len(contours1) > 5:
  #   # Fit ellipse to contour
  #   ellipse = cv2.fitEllipse(contours1)

  #   # Get angle of the fitted ellipse
  #   angle = ellipse[2]
  
  #   # print("Neck angle:", angle)
    
  #   if angle > 90:
  #     angle = 180 - angle 
    
  #   if np.abs(angle - ang_filtered1) > 30:
  #     print('Skipped')
  #   else:
  #     ang_filtered1 = ang_filtered1 * alpha + angle * (1 - alpha)
  #     print("Neck angle1:", ang_filtered1)  
  



  # cv2.imshow('det', dst)