import argparse
from src.models.segmentator.fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from src.models.segmentator.utils.tools import convert_box_xywh_to_xyxy
import glob
from tqdm import tqdm
import os


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    for img_path in tqdm(glob.glob(os.path.join(args.img_folder,"*.jpg"))):
        image_name = img_path.split("/")[-1]
        image_dir = "/".join(img_path.split("/")[:-2]) 
        output_dir = os.path.join(image_dir,args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)     

        input = Image.open(img_path)
        input = input.convert("RGB")
        everything_results = model(
            input,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou    
            )        
        bboxes = None
        points = None
        point_label = None
        prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
        if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
                ann = prompt_process.box_prompt(bboxes=args.box_prompt)
                bboxes = args.box_prompt
        elif args.text_prompt != None:
            ann = prompt_process.text_prompt(text=args.text_prompt)
        elif args.point_prompt[0] != [0, 0]:
            ann = prompt_process.point_prompt(
                points=args.point_prompt, pointlabel=args.point_label
            )
            points = args.point_prompt
            point_label = args.point_label
        else:
            ann = prompt_process.everything_prompt()
        prompt_process.plot(
            annotations=ann,
            output_path=os.path.join(output_dir, image_name),
            bboxes = bboxes,
            points = points,
            point_label = point_label,
            withContours=args.withContours,
            better_quality=args.better_quality,
        )


