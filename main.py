# from io import StringIO
from pathlib import Path
import streamlit as st
import time
import detect 
import val
import os
import glob
# import sys
import argparse
from PIL import Image
import torch
import cv2
import shutil
import yaml
import pandas as pd

def edit_yaml_file(new_dir):
    # Load the YAML file
    with open('data/test.yaml', 'r') as file:
        data = yaml.safe_load(file)
    # Replace the paths
    data['test'] = f'{new_dir}/images'
    data['val'] = f'{new_dir}/images'
    data['train'] = f'{new_dir}/images'
    # Write the changes back to the file
    with open('data/test1img.yaml', 'w') as file:
        yaml.dump(data, file)


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def main():
    # path = "runs\detect"
    # if os.path.exists(path):
    #     shutil.rmtree(path) 

    st.title('Object Recognition Dashboard')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-visdrone.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    # parser.add_argument("--data", type=str, default="data/visdrone.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default= "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--soft",default=False, action="store_true", help="use Soft-NMS")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.line_thickness = 1
    opt.hide_conf = True

    val_parser = argparse.ArgumentParser()
    val_parser.add_argument("--data", type=str, default="data/test1img.yaml", help="dataset.yaml path")
    val_parser.add_argument("--weights", nargs="+", type=str, default='weights/yolov5s-visdrone.pt', help="model path(s)")
    val_parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    val_parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    val_parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    val_parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    val_parser.add_argument("--task", default="test", help="train, val, test, speed or study")
    val_parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    val_parser.add_argument("--soft", action="store_true", default=None, help="use Soft-NMS")
    val_parser.add_argument("--save-hybrid", action="store_true", default=False, help="save label+prediction hybrid results to *.txt")
    val_opt = val_parser.parse_args()
    

    st.sidebar.title("Settings")
    
    source = ("image", "video")
    source_index = st.sidebar.selectbox("input", range(len(source)), format_func=lambda x: source[x])
    img_file = None
    vid_file = None
    if source_index == 0:
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
        if data_src == 'Sample data':
            img_path = glob.glob('data/images/sample_images/*')
            img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
            img_file = img_path[img_slider - 1]
            is_valid = True
        else:
            uploaded_file = st.sidebar.file_uploader(
                "upload image", type=['png', 'jpeg', 'jpg'])
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text='Uploading...'):
                    # st.sidebar.image(uploaded_file)
                    st.image(uploaded_file, caption="Selected Image")
                    img_name = str(uploaded_file.name)
                    img_name = img_name[:-4]
                    picture = Image.open(uploaded_file)
                    picture = picture.save(f'data/images/{uploaded_file.name}')
                    opt.source = f'data/images/{uploaded_file.name}'
                    edit_yaml_file(img_name)
            else:
                is_valid = False
    else:
        # data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
        # if data_src == 'Sample data':
        #     vid_file = 'data/videos/sample.mp4'
        #     opt.source = 'data/videos/sample.mp4'
        #     is_valid = True
        # else:
        uploaded_file = st.sidebar.file_uploader("upload video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Uploading...'):
                st.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                vid_file = f'data/videos/{uploaded_file.name}'
                opt.source = vid_file
                
        else:
            is_valid = False

    model_name_option = st.sidebar.selectbox("model", ("yolov5s", "yolov5-cus1", "orther"))
    
    model_weights = {
    "yolov5s": "weights/yolov5s-visdrone.pt",
    "yolov5-cus1": "weights/OursYOLO.pt",
    }
    
    if model_name_option in model_weights:
        opt.weights = model_weights[model_name_option]
        val_opt.weights = model_weights[model_name_option]
    else:
        uploaded_model = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if uploaded_model is not None:
            is_valid = True
            with st.spinner(text="Uploading..."):
                model_path = os.path.join("weights", uploaded_model.name)
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
            opt.weights = model_path
            val_opt.weights = model_path
        else:
            is_valid = False

    confidence = st.sidebar.slider('Confidence', min_value=0.001, max_value=1.0, value=.35)    
    opt.conf_thres = confidence
    val_opt.conf_thres = confidence
    # opt.conf_thres = 0.001
    
    iou = st.sidebar.slider('Iou', min_value=0.1, max_value=1.0, value=.45)
    opt.iou_thres = iou
    val_opt.iou_thres = iou
    # opt.iou_thres = 0.6

    if torch.cuda.is_available():
        # device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        # opt.device = 'CUDA' if device_option == 'cuda' else 'cpu'
        # val_opt.device = 'CUDA' if device_option == 'cuda' else 'cpu'
        dev = st.sidebar.text_input('DEVICE','cpu')
        opt.device = dev
        val_opt.device = dev
    else:
        # device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)
        opt.device = 'cpu'
    if st.sidebar.checkbox("Soft-NMS"):
        opt.soft = True
        val_opt.soft = True

    if is_valid:
        print('valid')
        print(opt)
        # if st.sidebar.button('detect'):
        if source_index == 0:
            with st.spinner(text='Preparing Images'):
                if img_file:
                    st.image(img_file, caption="Selected Image")
                    opt.source = str(img_file)
                    detect.main(opt)
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}')/ img), caption="Model prediction")
                else:
                    detect.main(opt)
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
                    val.main(val_opt)
                    df = pd.read_csv("data/result.csv")
                    st.write(df)
        # else:
            # if st.button('detect'):
            #     detect.main(opt)
            #     with st.spinner(text='Preparing Video'):
            #         for vid in os.listdir(get_detection_folder()):
                        # st.video(str(Path(f'{get_detection_folder()}') / vid))

                # try:
                #     cap = cv2.VideoCapture(vid_file)
                #     # custom_size = st.sidebar.checkbox("Custom frame size")
                #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #     # if custom_size:
                #     #     width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
                #     #     height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

                #     fps = 0
                #     st1, st2, st3 = st.columns(3)
                #     with st1:
                #         st.markdown("## Height")
                #         st1_text = st.markdown(f"{height}")
                #     with st2:
                #         st.markdown("## Width")
                #         st2_text = st.markdown(f"{width}")
                #     with st3:
                #         st.markdown("## FPS")
                #         st3_text = st.markdown(f"{fps}")
                #     st.markdown("---")
                #     output = st.empty()
                #     prev_time = 0
                #     curr_time = 0
                #     fr = 0
                #     while True:
                #         ret, frame = cap.read()
                #         if not ret:
                #             st.write("Can't read frame, stream ended? Exiting ....")
                #             break
                #         frame = cv2.resize(frame, (width, height))
                #         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, channels="RGB")
                #         frame_name = f'frame_{fr}.jpg' 
                #         temp = os.path.join('data','videos','temp',frame_name)
                #         fr = fr + 1
                #         status = cv2.imwrite(temp, frame) 
                #         # print(status)
                #         opt.source = temp
                #         detect.main(opt)
                #         for img in os.listdir(get_detection_folder()):
                #             output.image(str(Path(f'{get_detection_folder()}') / img))
                #         curr_time = time.time()
                #         fps = 1 / (curr_time - prev_time)
                #         prev_time = curr_time
                #         st1_text.markdown(f"**{height}**")
                #         st2_text.markdown(f"**{width}**")
                #         st3_text.markdown(f"**{fps:.2f}**")

                #     cap.release()
                # except:
                #     st.markdown("Stoped")
            
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass