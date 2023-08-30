# from flask import Flask
from ocr import *
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import os
from fastapi.responses import RedirectResponse

import uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get('/{filename}')
def hello(filename:str):
    # return {"Hello": "World"}
    filepath='\\'+"englis_ocr\my-app\static"

    filepath=filepath+'\\'+filename
    resized_image=extraction(filepath)
    grayscale_image=convert_image_to_grayscale(resized_image)
    thresholded_image=threshold_image(grayscale_image)
    inverted_image=invert_image(thresholded_image)
    dilated_image=dilate_image(inverted_image)
    contours=find_contours(dilated_image,resized_image)
    rectangular_contours=filter_contours_and_leave_only_rectangles(contours,resized_image)
    contour_with_max_area,max_area=find_largest_contour_by_area(rectangular_contours,resized_image)
    contour_with_max_area_ordered=order_points_in_the_contour_with_max_area(contour_with_max_area,resized_image)
    new_image_width,new_image_height=calculate_new_width_and_height_of_image(resized_image,contour_with_max_area_ordered)
    perspective_corrected_image=apply_perspective_transform(contour_with_max_area_ordered,new_image_width,new_image_height,resized_image)
    perspective_corrected_image_with_padding=add_10_percent_padding(resized_image,perspective_corrected_image)
    grayscale_image_with_padding=convert_image_to_grayscale(perspective_corrected_image_with_padding)
    thresholded_image_with_padding=threshold_image(grayscale_image_with_padding)
    inverted_image_with_padding=invert_image(thresholded_image_with_padding)
    dilated_image=dilate_image(inverted_image)
    vertical_lines_eroded_image=erode_vertical_lines(inverted_image_with_padding)
    horizontal_lines_eroded_image=erode_horizontal_lines(inverted_image_with_padding)
    combined_image=combine_eroded_images(vertical_lines_eroded_image,horizontal_lines_eroded_image)
    cell_contours=find_contours(combined_image,perspective_corrected_image_with_padding)
    bounding_boxes=convert_contours_to_bounding_boxes(cell_contours,perspective_corrected_image_with_padding,max_area)
    mean_height=get_mean_height_of_bounding_boxes(bounding_boxes)
    mean_width=get_mean_width_of_bounding_boxes(bounding_boxes)
    bounding_boxes=sort_bounding_boxes_by_y_coordinate(bounding_boxes)
    rows=club_all_bounding_boxes_by_similar_y_coordinates_into_rows(mean_height,bounding_boxes)
    rows=sort_all_rows_by_x_coordinate(rows)
    table=crop_each_bounding_box_and_ocr(perspective_corrected_image_with_padding,rows,mean_height,mean_width)
    # table=[[1,2,3,4],[1,2,3,4]]
    # return RedirectResponse("https://typer.tiangolo.com")

    return {"hello" : table }

# if __name__ == "__main__":
    # uvicorn.run(main, host="0.0.0.0", port=8000)
    # run(hello)

