import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import supervision as sv


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

# class LogoDetector:
#     def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model_id = model_id
        
#         self.processor = Owlv2Processor.from_pretrained(model_id)
#         self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
#         self.model.eval()
        
#     def run_example(self, image: Image, threshold=0.1, nms_threshold=0.3):
#         texts = [['a potential logo']]
#         inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
            
#         target_sizes = torch.Tensor([image.size[::-1]])
#         results = self.processor.post_process_object_detection(outputs=outputs, 
#                                                           target_sizes=target_sizes, 
#                                                           threshold=threshold,
#                                                         #   nms_threshold=nms_threshold
#                                                           )
        
#         scores = results[0]['scores'].cpu().numpy().reshape(-1, 1)
#         boxes = results[0]['boxes'].cpu().numpy()
        
#         detections = np.hstack((boxes, scores))
        
#         boxes_to_keep = sv.box_non_max_suppression(detections, iou_threshold=nms_threshold)
        
        
#         results = [{'scores': scores.flatten()[boxes_to_keep], 
#                     'boxes': boxes[boxes_to_keep],
#                     'labels': results[0]['labels'].cpu().numpy()[boxes_to_keep]
#                     }]
        
#         return results
    
#     def draw_boxes(self, image: Image, results):
#         visualized_image = image.copy()
#         draw = ImageDraw.Draw(visualized_image)
        
#         i = 0  # Retrieve predictions for the first image for the corresponding text queries
#         boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
#         logos = []
#         for box, score, label, color in zip(boxes, scores, labels, cycle(colormap)):
#             box = [round(i, 2) for i in box.tolist()]
#             x1, y1, x2, y2 = tuple(box)
#             draw.rectangle(xy=((x1, y1), (x2, y2)), width=5, outline=color)
#             logos.append(image.crop((x1, y1, x2, y2)))
#         return visualized_image, logos
    
#     def process_image(self, image: Image, threshold: float, nms_threshold: float):
#         results = self.run_example(image, threshold=threshold, nms_threshold=nms_threshold)
#         annotated_image, logos = self.draw_boxes(image, results)
#         return annotated_image, logos

class LogoDetector:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def run_example(self, image: Image, threshold=0.1, nms_threshold=0.3):
        texts = [['a potential logo']]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)


        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,
                                                          target_sizes=target_sizes,
                                                          threshold=threshold,
                                                        #   nms_threshold=nms_threshold
                                                          )

        scores = results[0]['scores'].cpu().numpy().reshape(-1, 1)
        boxes = results[0]['boxes'].cpu().numpy()

        detections = np.hstack((boxes, scores))

        boxes_to_keep = sv.box_non_max_suppression(detections, iou_threshold=nms_threshold)


        results = [{'scores': scores.flatten()[boxes_to_keep],
                    'boxes': boxes[boxes_to_keep],
                    'labels': results[0]['labels'].cpu().numpy()[boxes_to_keep]
                    }]

        return results

    def run_examples(self, images, threshold=0.1, nms_threshold=0.3):
        texts = [['a potential logo']*len(images)]
        inputs = self.processor(text=texts, images=images, return_tensors="pt").to(self.device)


        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1] for image in images])
        results = self.processor.post_process_object_detection(outputs=outputs,
                                                          target_sizes=target_sizes,
                                                          threshold=threshold,
                                                        #   nms_threshold=nms_threshold
                                                          )

        all_results = []
        for result in results:
          scores = result['scores'].cpu().numpy().reshape(-1, 1)
          boxes = result['boxes'].cpu().numpy()

          detections = np.hstack((boxes, scores))

          boxes_to_keep = sv.box_non_max_suppression(detections, iou_threshold=nms_threshold)


          all_results.append(
              {'scores': scores.flatten()[boxes_to_keep],
               'boxes': boxes[boxes_to_keep],
               'labels': result['labels'].cpu().numpy()[boxes_to_keep]
               })

        return results


    def draw_boxes(self, image: Image, results):
        # visualized_image = image.copy()
        # draw = ImageDraw.Draw(visualized_image)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        logos = []
        for box, score, label, color in zip(boxes, scores, labels, cycle(colormap)):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            # draw.rectangle(xy=((x1, y1), (x2, y2)), width=5, outline=color)
            logos.append(image.crop((x1, y1, x2, y2)))
        return logos, scores

    def process_image(self, image: Image, threshold: float, nms_threshold: float):
        results = self.run_example(image, threshold=threshold, nms_threshold=nms_threshold)
        annotated_image, logos, scores = self.draw_boxes(image, results)
        return annotated_image, logos, scores

    def process_images(self, images, threshold, nms_threshold, batch_size=2):
      all_results = []
      for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        results = self.run_examples(batch, threshold=threshold, nms_threshold=nms_threshold)
        for (image, result) in zip(batch, results):
          logos, scores = self.draw_boxes(image, [result])
          all_results.append({
            #   'image': annotated_images,
              'logos': logos,
              'scores': scores.cpu().numpy()
          })
      return all_results