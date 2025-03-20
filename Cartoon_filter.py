import cv2
import numpy as np

def cartoonify_with_strong_colors_and_bold_edges(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 800))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7) 
    
    edges1 = cv2.Canny(gray, 10, 200) 
    edges2 = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)  
    edges3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)
    edges = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=2) 
    edges = cv2.erode(edges, np.ones((5,5), np.uint8), iterations=1)  
    edges = cv2.bitwise_not(edges)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    color = cv2.bilateralFilter(img, 9, 150, 150) 
    color = cv2.stylization(color, sigma_s=200, sigma_r=0.5)  
    
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255) 
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.3, 0, 255)  
    enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    enhanced_img = cv2.convertScaleAbs(enhanced_color, alpha=2.0, beta=50) 
    
    cartoon = cv2.addWeighted(enhanced_img, 0.8, edges_colored, 0.7, 0)
    
    cv2.imshow("Strong Colors and Bold Edges Cartoon", cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cartoon

cartoonify_with_strong_colors_and_bold_edges("image.jpg")
