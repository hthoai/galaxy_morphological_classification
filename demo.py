import streamlit as st
import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa
import plotly.graph_objects as go

import torch
from torchvision.transforms import ToTensor

from lib.config import Config
from lib.experiment import Experiment


EXPERIMENT_NAME = "galaxy_0224-0528"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title("🌌 DEMO GALAXY CLASSIFICATION 🌌")

    run_the_app()


def run_the_app():
    st.sidebar.header("Input an image")
    file_format = ["png", "jpg", "jpeg"]
    uploaded_file = st.sidebar.file_uploader("Select a photo", type=file_format)

    if uploaded_file is None:
        return

    # st.write(uploaded_file.name)

    img_input = process_input(uploaded_file)

    st.write("Input galaxy image")
    st.image(img_input, channels="BGR", use_column_width=True)

    image_to_model = prepare_image(img_input)

    y_hat = detect_lane(image_to_model)
    place_holder = st.empty()

    draw_plot(y_hat, place_holder, uploaded_file.name.split('.')[0])


def process_input(input_file):
    file_bytes = np.asarray(input_file.getbuffer(), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return opencv_image


def prepare_image(img_input):
    # img = img_input / 255.0
    # img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img_input

    totensor = ToTensor()
    transformations = iaa.Sequential(
        [iaa.CropToFixedSize(207, 207), iaa.Resize({"height": 69, "width": 69})]
    )
    transform = iaa.Sequential([transformations])
    img = transform(image=img.copy())

    return totensor(img.astype(np.float32))


def detect_lane(img):
    @st.cache(allow_output_mutation=True)
    def load_network():
        exp = Experiment(EXPERIMENT_NAME, mode="test")
        cfg = Config(exp.cfg_path)
        exp.set_cfg(cfg, override=False)
        model = cfg.get_model()
        model.load_state_dict(exp.get_epoch_model(30))
        model = model.to(torch.device("cuda"))
        model.eval()

        return model

    classifier = load_network()
    with torch.no_grad():
        output = classifier(img[None, ...].cuda())
        output = torch.flatten(output).tolist()

        return output



def draw_plot(y_h, place_holder, img_id):
    df = pd.read_csv('datasets/training_solutions_rev1.csv')
    
    class_name = list(df.columns[1:])
    predicted = dict(zip(class_name, y_h))

    sorted_dict = dict(sorted(predicted.items(), key=lambda item: item[1]))
    class_name = list(sorted_dict.keys())
    y_hat = list(sorted_dict.values())
    gt = df[df.GalaxyID == int(img_id)][class_name].values[0][1:]

    gt_colors = ['green' if (gt[i] + y_hat[i]) < 0.01 else 'red' for i in range(len(gt))]
    # 
    fig = go.Figure(data=[
        go.Bar(
            y=class_name,
            x=y_hat,
            marker_color=y_hat,
            orientation="h",
            name='Predicted'),
        go.Scatter(x=gt, y=class_name,
            # marker_color='rgb(255,0,0)',
            marker_color = gt_colors,
            name='Ground Truth',
            mode='markers',
            marker_size=9
        )
    ])

    fig.update_layout(
        # title_text="Least Used Feature",
        xaxis=dict(title="Probability", tickfont_size=14),
        yaxis=dict(titlefont_size=19, tickfont_size=17, tickmode='linear'),
        autosize=False,
        height=1000
    )

    place_holder.write(fig)


if __name__ == "__main__":
    main()
