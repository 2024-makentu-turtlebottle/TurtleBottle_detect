import numpy as np
import queue   

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
        if dst[next_pos] == 0 and calculate_distance(img[next_pos], img[current]) < 12:
          dst[next_pos] = 1
          q.put(next_pos)

  return dst


def get_edge(img):
  height, width = 80, 60
  edgesL = []
  for i in range(height):
    for j in range(width - 1):
      if img[i, j] == 0 and img[i, j+1] == 1:
        edgesL.append([j, i])
        break
      
  return np.array(edgesL)


import cv2
import time

cap = cv2.VideoCapture(0)

ang_filtered = 0
ang_filtered1 = 1
alpha = 0.6

index = 0
while(True):
  ret, frame = cap.read()
  frame = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
  frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
  
  dst = bfs(frame, (frame.shape[0]//2-1, frame.shape[1]//2-1))
  
  edgesL = get_edge(dst)
  
  cv2.imwrite(f"imgs/frame_{index}.jpg", frame)
  index += 1
  
  if len(edgesL) > 0:
    cv2.drawContours(frame, [edgesL], -1, (0, 0, 255), 2)    
  if edgesL.shape[0] > 5:
    x, y = edgesL[:,0], edgesL[:,1]
    A = np.vstack([y, np.ones(len(y))]).T
    
    AT = A.T
    ATMA = AT @ A
    
    ATMAI = np.linalg.inv(ATMA)
    
    ATMAIAT = ATMAI @ AT
    
    ATMAIATx = ATMAIAT @ x
    
    print(ATMAIATx[0])

  # Show image
  cv2.imshow("Contours", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  time.sleep(0.1)

cap.release()

cv2.destroyAllWindows()