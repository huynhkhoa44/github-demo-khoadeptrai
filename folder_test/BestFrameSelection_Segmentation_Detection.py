import cv2
import os
import numpy as np
import pydicom as dicom
from ultralytics import YOLO
from PIL import Image, ImageSequence


class YOLO_Pipeline:
    def __init__(self):
        self.classification_model = YOLO('cls_80epoch.pt')
        self.segmentation_model = YOLO('anatomic_segmentation_71.pt')
        self.lesion_detection_model = YOLO('lesion_detection_15epoch.pt')

    def process_dicom(self, dicom_file, output_path):
        best_frame = self.extract_best_frame(dicom_file)
        if best_frame is not None:
            segmentation_result = self.segment_frame(best_frame)
            self.save_result(segmentation_result, output_path)
            self.detect_lesions(os.path.join(output_path, 'best_segmented_frame.png'))
        else:
            print("No suitable high-quality frame found.")

    def extract_best_frame(self, dicom_file):
        ds = dicom.dcmread(dicom_file)
        pixel_array = ds.pixel_array

        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array
        elif ds.PhotometricInterpretation == "YBR_FULL":
            pixel_array = np.frombuffer(ds.PixelData, dtype=np.uint8).reshape(ds.Rows, ds.Columns, 3)
        pixel_array = pixel_array.astype(np.uint8)

        best_frame = None
        best_confidence = -1

        for j in range(pixel_array.shape[0]):
            slice = pixel_array[j]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_slice = clahe.apply(slice)

            enhanced_slice_rgb = cv2.cvtColor(enhanced_slice, cv2.COLOR_GRAY2RGB)

            results = self.classification_model.predict(enhanced_slice_rgb, imgsz=512, show_labels=False,
                                                        show_boxes=False)
            confidence = results[0].probs.top1conf.item()

            if int(results[0].probs.top1) == 0 and confidence > best_confidence:
                best_frame = enhanced_slice
                best_confidence = confidence

        return best_frame

    def segment_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        results = self.segmentation_model.predict(frame_rgb, imgsz=512, conf=0.1, show_labels=False, show_boxes=False)
        segmented_frame = self.textAndContour_segment(frame, results)
        return segmented_frame

    def textAndContour_segment(self, img, results):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        check_lumen = 0

        if results[0].masks is not None:
            for idx, prediction in enumerate(results[0].boxes.xywhn):
                class_id_int = int(results[0].boxes.cls[idx].item())
                poly = results[0].masks.xyn[idx].tolist()
                poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)
                poly *= [w, h]

                if class_id_int == 0 and check_lumen == 0:
                    cv2.polylines(img, [poly.astype('int')], True, (255, 0, 0), 1)
                    check_lumen += 1
                elif class_id_int == 1:
                    cv2.polylines(img, [poly.astype('int')], True, (0, 255, 0), 1)
                elif class_id_int == 2:
                    cv2.polylines(img, [poly.astype('int')], True, (0, 0, 255), 1)

        return img

    def save_result(self, segmented_frame, output_path):
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, 'best_segmented_frame.png'), segmented_frame)

    def detect_lesions(self, segmented_image_path):
        results_detect = self.lesion_detection_model.predict(segmented_image_path, imgsz=512, conf=0.01,
                                                             show_labels=True)

        annotated_frame_detect = results_detect[0].plot(labels=True)

        cv2.imshow("Lesion Detection Result", annotated_frame_detect)

        # Save the lesion detection result
        output_dir = os.path.dirname(segmented_image_path)
        cv2.imwrite(os.path.join(output_dir, 'lesion_detection_result.png'), annotated_frame_detect)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()


# Usage
pipeline = YOLO_Pipeline()
pipeline.process_dicom('LAD_anon.DCM', 'RESULT_PIPELINE_1/')