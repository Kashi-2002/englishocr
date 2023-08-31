import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import os
import torch



def extraction(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    scale_percent = 20
    width = int(img.shape[1]  + (img.shape[1] * scale_percent / 100))
    height = int(img.shape[0]  +(img.shape[0] * scale_percent / 100))
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def convert_image_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def threshold_image(grayscale_image):
    thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def invert_image(thresholded_image):
    inverted_image = cv2.bitwise_not(thresholded_image)
    return inverted_image

def dilate_image(inverted_image):
    dilated_image = cv2.dilate(inverted_image, None, iterations=5)
    return dilated_image

def find_contours(dilated_image,image):
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Below lines are added to show all contours
    # This is not needed, but it is useful for debugging
    # image_with_all_contours = image.copy()
    # cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
    return contours

def filter_contours_and_leave_only_rectangles(contours,image):
    rectangular_contours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            rectangular_contours.append(approx)
    # Below lines are added to show all rectangular contours
    # This is not needed, but it is useful for debugging
    # image_with_only_rectangular_contours = image.copy()
    # cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
    return rectangular_contours

def find_largest_contour_by_area(rectangular_contours,image):
    max_area = 0
    contour_with_max_area = None
    for contour in rectangular_contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            contour_with_max_area = contour
    # Below lines are added to show the contour with max area
    # This is not needed, but it is useful for debugging
    # image_with_contour_with_max_area = image.copy()
    # cv2.drawContours(image_with_contour_with_max_area, [contour_with_max_area], -1, (0, 255, 0), 3)
    return contour_with_max_area,max_area


def order_points_in_the_contour_with_max_area(contour_with_max_area,image):
    contour_with_max_area_ordered = order_points(contour_with_max_area)
    # The code below is to plot the points on the image
    # it is not required for the perspective transform
    # it will help you to understand and debug the code
    # image_with_points_plotted = image.copy()
    # for point in contour_with_max_area_ordered:
    #     point_coordinates = (int(point[0]), int(point[1]))
    #     image_with_points_plotted = cv2.circle(image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
    return contour_with_max_area_ordered

def calculate_new_width_and_height_of_image(image,contour_with_max_area_ordered):
    existing_image_width = image.shape[1]
    existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)

    distance_between_top_left_and_top_right = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[1])
    distance_between_top_left_and_bottom_left = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[3])

    aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

    new_image_width = existing_image_width_reduced_by_10_percent
    new_image_height = int(new_image_width * aspect_ratio)

    return new_image_width,new_image_height

def apply_perspective_transform(contour_with_max_area_ordered,new_image_width,new_image_height,image):
    pts1 = np.float32(contour_with_max_area_ordered)
    pts2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_corrected_image = cv2.warpPerspective(image, matrix, (new_image_width, new_image_height))
    return perspective_corrected_image

# Below are helper functions
def calculateDistanceBetween2Points(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def add_10_percent_padding(image,perspective_corrected_image):
    image_height = image.shape[0]
    padding = int(image_height * 0.1)
    perspective_corrected_image_with_padding = cv2.copyMakeBorder(perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return perspective_corrected_image_with_padding


def erode_vertical_lines(inverted_image):
    hor = np.array([[1,1,1,1,1,1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=25)
    return (vertical_lines_eroded_image)


def erode_horizontal_lines(inverted_image):
    ver = np.array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=20)
    return (horizontal_lines_eroded_image)


def combine_eroded_images(vertical_lines_eroded_image,horizontal_lines_eroded_image):
    combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    return combined_image


def find_contours(dilated_image,original_image):
    result = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]
    # The code below is for visualization purposes only.
    # It is not necessary for the OCR to work.
    image_with_contours_drawn = original_image.copy()
    cv2.drawContours(image_with_contours_drawn, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("\englis_ocr\my-app\static\img_10.jpg",image_with_contours_drawn)
    return contours


def convert_contours_to_bounding_boxes(contours,original_image,max_area):
    bounding_boxes = []
    image_with_all_bounding_boxes = original_image.copy()
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if(w*h<max_area):
        bounding_boxes.append((x, y, w, h))
        # image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # cv2_imshow(image_with_all_bounding_boxes)
    return bounding_boxes

def get_mean_height_of_bounding_boxes(bounding_boxes):
    heights = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        heights.append(h)
    return np.mean(heights)

def get_mean_width_of_bounding_boxes(bounding_boxes):
    heights = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        heights.append(w)
    return np.mean(heights)


def sort_bounding_boxes_by_y_coordinate(bounding_boxes):
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
    return bounding_boxes

def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(mean_height,bounding_boxes):
    rows = []
    half_of_mean_height = mean_height / 2
    current_row = [ bounding_boxes[0] ]
    for bounding_box in bounding_boxes[1:]:
        current_bounding_box_y = bounding_box[1]
        previous_bounding_box_y = current_row[-1][1]
        distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
        if distance_between_bounding_boxes <= half_of_mean_height:
            current_row.append(bounding_box)
        else:
            rows.append(current_row)
            current_row = [ bounding_box ]
    rows.append(current_row)
    return rows

def sort_all_rows_by_x_coordinate(rows):
    for row in rows:
        row.sort(key=lambda x: x[0])
    return rows


def getSkewAngle(cvImage):
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotation(image,filename):
    angle_degrees = getSkewAngle(image)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, scale=1)
    new_width = int(width * np.abs(np.cos(np.radians(angle_degrees))) + height * np.abs(np.sin(np.radians(angle_degrees))))
    new_height = int(height * np.abs(np.cos(np.radians(angle_degrees))) + width * np.abs(np.sin(np.radians(angle_degrees))))
    tx = (new_width - width) / 2
    ty = (new_height - height) / 2
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),borderValue=(255,255,255))
    white_background = np.ones_like(rotated_image) * 255
    result = cv2.add(white_background, rotated_image)
    cv2.imwrite(filename,result)



def crop_each_bounding_box_and_ocr(original_image,rows,mean_height,mean_width):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    table = []
    current_row = []
    image_number = 0
    for row in rows:
        for bounding_box in row:
            x, y, w, h = bounding_box
            # y = y - 5
            if(h>=mean_height or w>=mean_width):
                cropped_image = original_image[y:y+h, x:x+w]
                image_slice_path = "\englis_ocr\my-app\static\img_" + str(image_number) + ".jpg"
                cv2.imwrite(image_slice_path, cropped_image)
                image = Image.open(image_slice_path).convert("RGB")
                pixel_values = processor(image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values, output_scores=True,return_dict_in_generate=True)
                generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
                composed = torch.cat([preds.flatten() if preds is not None else torch.empty(0, dtype=torch.float32) for preds in generated_ids.scores], dim=-1)
                logsumexp = torch.logsumexp(composed, dim=-1, keepdim=True)
                if(logsumexp.numpy()[0]<float(20)):
                    img = cv2.imread(image_slice_path, cv2.IMREAD_COLOR)
                    scale_percent = 100
                    width = int(img.shape[1]  + (img.shape[1] * scale_percent / 100))
                    height = int(img.shape[0]  +(img.shape[0] * scale_percent / 100))
                    dsize = (width, height)
                    output = cv2.resize(img, dsize)
                    rotation(output,image_slice_path)
                    pixel_values = processor(img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            current_row.append(generated_text)
            image_number += 1
            os.remove(image_slice_path)
        table.append(current_row)
        current_row = []
    return table
