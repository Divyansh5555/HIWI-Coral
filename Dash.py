import os
import base64
import cv2
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import json
from dash_canvas import DashCanvas
import base64
import numpy as np
import torchvision.transforms as T
import dash
from chatbot_component import get_chatbot_ui
from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from PIL import Image
from io import BytesIO
from deeplabv3pp_segmentation import DeepLabV3PlusCustom
from dash.exceptions import PreventUpdate
from dash.dcc import send_file
import zipfile
import csv

def encode_np_image(np_img):
    pil_img = Image.fromarray(np_img.astype("uint8"))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"

def load_image_tensor(image_path, size=(256, 256)):
    image = Image.open(image_path).convert("RGB").resize(size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def load_raw_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def load_model(path):
    model = DeepLabV3PlusCustom()
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_mask(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)[0]
        output = torch.sigmoid(output)
        pred = output.squeeze().numpy()
        print("mask", ((pred > 0.5).astype(np.uint8) * 255).shape)
        return (pred > 0.5).astype(np.uint8) * 255

def overlay_mask_transparent(image, mask, color=(255, 0, 0), alpha=0.4):
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = color[0]
    red_mask[:, :, 1] = color[1]
    red_mask[:, :, 2] = color[2]
    overlay = np.where(mask[:, :, None] > 0, (1 - alpha) * image + alpha * red_mask, image)
    return overlay.astype(np.uint8)

def create_overlay_with_numbers(image, mask, contours, module ,color=(255, 0, 0), alpha=0.4):
    """
    Creates an overlay of the mask on the image, and draws numbers at the center of each contour.

    Args:
        image (np.array): The original image (H, W, 3).
        mask (np.array): The binary mask (H, W).
        contours (list): A list of contours found in the mask.
        color (tuple): RGB color for the overlay (default: red).
        alpha (float): Transparency level for the overlay (default: 0.4).

    Returns:
        np.array: The image with the transparent mask overlay and contour numbers.
    """
    # First, create the base transparent overlay
    overlay_image = overlay_mask_transparent(image, mask,color=color, alpha=alpha)

    # Convert the overlay to BGR if it's not already (OpenCV typically works with BGR)
    # Assuming image is RGB, overlay_mask_transparent returns RGB.
    # We might need to convert to BGR for cv2.putText if not handled internally.
    # For simplicity, let's assume overlay_image is already suitable for cv2.putText or
    # that cv2.putText handles RGB correctly (it does for putting text directly).

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 10
    text_color = (0, 0, 0) # Black color for the numbers
    
    if module.lower() == "polyp":
        font_scale = 0.8
        font_thickness = 1
        for i, contour in enumerate(contours):
            print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 0:
            # Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # If the moment is zero (e.g., a single point contour), use the first point
                    cX, cY = contour[0][0]
                    
        
                # Draw the number on the overlay image
                text = str(i + 1) # Numbers start from 1
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = cX - text_size[0] // 2
                text_y = cY + text_size[1] // 2 # Adjust y to center vertically
        
                cv2.putText(overlay_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    else:
        for i, contour in enumerate(contours):
           # print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 2000:
            # Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # If the moment is zero (e.g., a single point contour), use the first point
                    cX, cY = contour[0][0]
                    
        
                # Draw the number on the overlay image
                text = str(i + 1) # Numbers start from 1
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = cX - text_size[0] // 2
                text_y = cY + text_size[1] // 2 # Adjust y to center vertically
        
                cv2.putText(overlay_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
      
    return overlay_image.astype(np.uint8)

def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_mask_and_contours(module):
    import tempfile
    mask = CURRENT[module]['mask']
    contours = CURRENT[module]['contours']
    temp_dir = tempfile.mkdtemp()
    mask_path = os.path.join(temp_dir, f"{module}_mask.png")
    csv_path = os.path.join(temp_dir, f"{module}_contours.csv")

    cv2.imwrite(mask_path, mask)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Contour_ID', 'X', 'Y'])
        for i, cnt in enumerate(contours):
            for pt in cnt:
                x, y = pt[0]
                writer.writerow([i, x, y])

    zip_path = os.path.join(temp_dir, f"{module}_export.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(mask_path, os.path.basename(mask_path))
        zipf.write(csv_path, os.path.basename(csv_path))
    return zip_path

IMAGE_DIRS = {
    "Polyp": "/Users/mikan/Downloads/Evaluation/project/test/images",
    "Coral": "/Users/mikan/Downloads/Evaluation/project/test/coral_images"
}
MODEL_PATHS = {
    "Polyp": "/Users/mikan/Downloads/Evaluation/project/test/models/deeplabv3pp_polypsnew_seed.pth",
    "Coral": "/Users/mikan/Downloads/Evaluation/project/test/models/unetpp_coral.pth"
}
CURRENT = {"Polyp": {}, "Coral": {}}

app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO], suppress_callback_exceptions=True)
app.title = "Coral & Polyp Segmentation Dashboard"

def create_module_tab(module):
    image_list = [f for f in os.listdir(IMAGE_DIRS[module]) if f.lower().endswith(('.jpg', '.png'))][:6]

    return dbc.Tab(label=f"{module} Segmentation", tab_id=module, children=[
        dbc.Row([
            # LEFT COLUMN: Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"{module} Image & Model"),
                    dbc.CardBody([
                        html.P("Choose image, run prediction, and quantify segments.", className="text-light small"),

                        html.Label("Select Image:", className="mt-2"),
                        html.Div(
                            children=[
                                html.Img(
                                    src=encode_np_image(load_raw_image(os.path.join(IMAGE_DIRS[module], img))),
                                    id={"type": "thumb", "index": img, "module": module},
                                    n_clicks=0,
                                    style={
                                        "height": "80px", "width": "auto", "marginRight": "8px",
                                        "cursor": "pointer", "border": "1px solid #ccc", "borderRadius": "5px"
                                    }
                                ) for img in image_list
                            ],
                            style={
                                "display": "flex", "overflowX": "auto", "whiteSpace": "nowrap",
                                "paddingBottom": "10px", "gap": "8px", "scrollbarWidth": "thin"
                            }
                        ),

                        dbc.Button("Run Prediction",
                                   id={"type": "predict-button", "module": module},
                                   color="primary", className="mt-3 mb-2", size="sm"),

                        dcc.Loading(
                            id={'type': 'loading-wrapper', 'module': module},
                            type="default", color="none",
                            children=html.Div(id={'type': 'loading-msg', 'module': module}),
                            style={"marginTop": "20px", "textAlign": "center"},
                            fullscreen=False
                        ),

                        html.Hr(),

                        html.Label("Select Mask for Quantification:", style={'margin-bottom': '10px'}),
                        dcc.Dropdown(id={"type": "contour-dropdown", "module": module},
                                     placeholder="Select Mask", style={"color": "black"}),

                        html.Div(id={"type": "quant-result", "module": module}, className="text-warning mb-3"),

                        html.Div([
                            dbc.Button("Download Mask + CSV",
                                       id={"type": "download-button", "module": module},
                                       color="success", size="sm"),
                            dcc.Download(id={"type": "download-component", "module": module})
                        ], className="mt-3", style={"textAlign": "center"}),

                        #  Add dummy store for mask-base64 for all modules
                        dcc.Store(id={"type": "mask-base64", "module": module})
                    ])
                ])
            ], width=3),

            # RIGHT COLUMN: Tabs
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Tabs(
                            children=[
                                dbc.Tab(label="Original Image", tab_id=f"tab-original-{module}", children=[
                                    html.Div(id={"type": "original-image", "module": module}, className="mt-3")
                                ]),
                                dbc.Tab(label="Predicted Mask", tab_id=f"tab-mask-{module}", children=[
                                    html.Div(id={"type": "pred-mask", "module": module}, className="mt-3")
                                ]),
                                dbc.Tab(label="Selected Mask", tab_id=f"tab-contour-{module}", children=[
                                    html.Div(id={"type": "contour-mask", "module": module}, className="mt-3")
                                ]),
                                dbc.Tab(label="Masked Images", tab_id=f"tab-masks-{module}", children=[
                                    html.Div(id={"type": "mask-icons", "module": module}),
                                    dcc.Store(id={"type": "mask-metadata", "module": module}),
                                    html.Div(id={"type": "mask-info-box", "module": module}, className="mt-2")
                                ]),
                                dbc.Tab(label="Evaluations", tab_id=f"tab-groundtruth-{module}", children=[
                                    html.Div(id={'type': 'ground-truth', 'module': module}),
                                    html.Div(id={'type': 'distance-graph', 'module': module}, className="mt-3")
                                ]),

                                #  Only Coral gets the Metrics tab
                                *([
                                    dbc.Tab(label="Metrics", tab_id=f"tab-metrics-{module}", children=[
                                        html.P("Draw a straight line on the scale/ruler (default: 5mm).", className="text-light"),

                                        html.Div(
                                            children=html.Img(id={'type': 'metrics-image', 'module': module},
                                                              style={'width': '100%', 'maxWidth': '512px'}),
                                            className="mb-3"
                                        ),

                                        DashCanvas(
                                            id={'type': 'scale-canvas', 'module': module},
                                            width=512,
                                            height=512,
                                            lineWidth=3,
                                            tool="line",
                                            lineColor="red",
                                            hide_buttons=['zoom', 'pan', 'reset', 'save', 'pencil', 'select', 'rectangle', 'circle', 'polygon'],
                                            goButtonTitle='Done',
                                            filename='',
                                            image_content=""
                                        ),

                                        dbc.Button("Calculate ImageJ Metrics", id={'type': 'calc-btn', 'module': module},
                                                   className="btn btn-info btn-sm mt-3"),

                                        html.Div(id={'type': 'imagej-report', 'module': module}, className="text-light mt-2")
                                    ])
                                ] if module.lower() == "coral" else [])
                            ],
                            id={"type": "child-tabs", "module": module},
                            active_tab=f"tab-original-{module}"
                        )
                    ], width=10)
                ])
            ], width=9)
        ])
    ])

@app.callback(
    Output({'type': 'metrics-image', 'module': MATCH}, 'src'),
    Input({'type': 'mask-base64', 'module': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_metrics_image(mask_base64):
    triggered_id = ctx.triggered_id
    if not triggered_id or "module" not in triggered_id:
        raise PreventUpdate

    module = triggered_id["module"]

    # 🚨 Skip this callback for non-Coral modules
    if module.lower() != "coral":
        raise PreventUpdate

    if not mask_base64:
        return None

    return f"data:image/png;base64,{mask_base64}"
@app.callback(
    Output({'type': 'mask-icons', 'module': MATCH}, 'style'),
    Input({'type': 'child-tabs', 'module': MATCH}, 'active_tab'),
    State({'type': 'child-tabs', 'module': MATCH}, 'id')
)
def show_only_on_mask_tab(active_tab, tab_id):
    return {"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginTop": "10px"} \
        if active_tab == f"tab-masks-{tab_id['module']}" else {"display": "none"}    
    
app.layout = dbc.Container(fluid=True, children=[

    # ---------------------- MAIN LAYOUT ----------------------
    dbc.Row([

        # Sidebar Navigation
        dbc.Col([
            html.Div([
                html.H4("Menu", className="text-white mb-4"),
                dbc.Nav([
                    dbc.NavLink("Home", href="#hero", external_link=True),
                    dbc.NavLink("Coral Segmentation", href="#coral", external_link=True),
                    dbc.NavLink("Polyp Segmentation", href="#polyp", external_link=True),
                    dbc.NavLink("Help", href="#help", external_link=True)
                ], vertical=True, pills=True, className="sidebar-nav")
            ], className="sidebar p-4")
        ], width=2),

        # Content Panels
        dbc.Col([

            # ---------------------- Banner Image ----------------------
            html.Div([
                dcc.Interval(id="banner-interval", interval=4000, n_intervals=0),
                dcc.Store(id="banner-index", data=0),
                html.Img(
                    id="banner-img",
                    src="/assets/banner1.jpg",  # initial banner
                    style={
                        "width": "95%",
                        "height": "300px",
                        "alignSelf": "center",
                        "objectFit": "cover",
                        "borderRadius": "16px",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.4)",
                        "marginTop": "20px",
                        "marginBottom": "30px"
                    }
                )
            ], style={"align-self": "center", "margin-left": "50px"}),

            # ---------------------- HERO / HOME SECTION ----------------------
            html.Div(id="hero", children=[
                html.Div([
                    html.H1("Click, Segment, Measure: Coral Morphology in One Place", className="text-info mb-4 "),

                    html.Div([
                        html.P(
                            "This interactive web tool was created to help researchers  detect and analyze coral and polyp structures from 2D lab-captured images. "
                            "By focusing on the key morphological traits like shape, size, and surface coverage it supports studies that aim to understand how these organisms grow, change, and interact with their environment.",
                            className="lead text-secondary"
                        ),
                       html.P(
                            "We used DeepLabV3++ for polyp segmentation and U-Net++ for coral segmentation. "
                            "The polyp model achieved a Dice score of 0.5530 and IoU of 0.3832. "
                            "The coral model reached a Dice score of 0.9707 and IoU of 0.9438, ensuring high accuracy in segmenting complex coral structures.",
                              className="lead text-secondary"

                        ),
                        html.P(
                            "In simple terms, the Dice score and IoU are two ways of measuring how closely the predicted area matches the actual region we're trying to identify. "
                            "A Dice score near 1 means the model’s prediction overlaps really well with the true mask, while IoU measures the shared space between the prediction and ground truth. "
                            "Both metrics give a sense of how reliable the segmentation is. Inference time just tells us how fast the model runs—lower is better, and here, it’s impressively fast.",
                            className="lead text-secondary"
                        ),
                        html.P(
                            "For polyp segmentation, the task was significantly harder. Polyps come in all shapes and sizes—they overlap, vary in ratio, and often blend into the background. "
                            "Despite this complexity, the model still achieved a Dice score of 0.5530 and an IoU of 0.3832. While these numbers are lower than for corals, they are still strong "
                            "considering the challenge, and notably better than what other models could achieve under the same conditions. The inference time remained around 0.0191 seconds—"
                            "so the speed and responsiveness of the tool were never compromised.",
                            className="lead text-secondary"
                        ),
                        html.P(
                            "What makes this tool truly helpful is its ability to go beyond just segmenting the images. It calculates the number of masks, their pixel counts, and even their physical area in millimeters squared, "
                            "based on the image scale. This allows researchers to trace coral or polyp growth over time and understand spatial patterns that might indicate health or growth .",
                            className="lead text-secondary"
                        ),
                        html.P(
                            "In the end, this semi-automated system shows how we can combine deep learning and thoughtful interface design to analyze coral morphology more efficiently. "
                            "It enables detailed analysis, quick visualizations, and user-defined trait recognition making it easier for scientists and marine biologists to study reef health, one image at a time."
                            "To know more about the tool, click the button below.",
                            className="lead text-secondary"
                        )
                    ]),

                    dbc.Button("Get insights about the tool", id="open-modal", color="primary", className="btn btn-primary btn-block mt-3"),

                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("Welcome to Coral and Polyp Analysis")),
                            dbc.ModalBody([
                                html.H5("Abstract", className="mt-2"),
                                html.Ul([
                                    html.Li("This thesis introduces a semi-automated web-based tool for coral morphology analysis using deep learning and human input."),
                                    html.Li("Coral regions are segmented with U-Net++, while individual polyps are detected using DeepLabV3++, enabling effective two-stage segmentation."),
                                    html.Li("Each region is annotated with mask numbers for direct measurement of area, count, and inter-polyp distances."),
                                    html.Li("A built-in scale calibration feature ensures accurate mm-based quantification of traits from pixel-based masks."),
                                    html.Li("The tool enables scalable, consistent, and low-latency coral trait tracking, supporting ecological research and reef health monitoring.")
                            ]),
                                html.H5("Key Highlights of the Semi-Automated Coral Morphometric Analysis Tool:", className="mt-2"),
                                html.Ul([
                                    html.Li("Scalable and Accessible Pipeline: The tool supports large-scale coral morphology analysis through a 2D lab-based approach, capturing diverse coral and polyp structures to reflect real-world biological variation."),
                                    html.Li("Quantitative Trait Extraction: Morphological features such as coral area, polyp size, and polyp count are accurately measured using pixel-based segmentation, translated into real-world metrics using calibrated scales (e.g., mm/px)."),
                                    html.Li("Benchmarking Modern Segmentation Models: The system evaluates state-of-the-art deep learning models (U-Net,U-Net++ , SAM, DeepLabV3++, SAM2) for coral segmentation, providing researchers with comparative performance metrics."),
                                    html.Li("User-Centric & Sustainable Design: The semi-automated interface is built for ease of use, minimizing manual effort while supporting scientific rigor — making it ideal for coral monitoring, trait analysis, and conservation workflows."),
                                ]),
                                html.H5("What this tool offers:", className="mt-2"),
                                html.Ul([
                                    html.Li("Load coral or polyp images for analysis"),
                                    html.Li("Use state-of-the-art models to segment coral and polyp regions"),
                                    html.Li("Measure pixel count, area, and other metrics"),
                                    html.Li("Download masks and reports in CSV format"),
                                ]),
                                html.H5("How to begin:", className="mt-4"),
                                html.Ul([
                                    html.Li("1 Click an image thumbnail and choose an image to analyze"),
                                    html.Li("2️ Press 'Run Prediction' to generate masks"),
                                    html.Li("3️ Wait for the model to process the image"),
                                    html.Li("4️ View segmented masks and choose predicted masks for quantification"),
                                    html.Li("5️ Explore the results and download masks in CSV format"),
                                ]),
                                html.H5("Quantification Formulas:", className="mt-4"),
                                html.Ul([
                                    html.Li("Pixel Count = Total number of pixels within the selected mask region."),
                                    html.Li("Known Scale = Real-world length (e.g., 5 mm) ÷ Pixel length (in pixels)."),
                                    html.Li("Pixel Area = (Pixel size in mm)² (i.e., mm²/pixel)."),
                                    html.Li("Region Area = Pixel Count × Pixel Area."),
                                    html.Li("Aspect Ratio = Image Width (px) ÷ Image Height (px).")
                                ])
                            ]),
                            dcc.Store(id="selected-tab-store", data="Home"),
                            dbc.ModalFooter(
                                dbc.Button("Start Now", href="#polyp", id="start-now-btn", color="success")
                            ),
                        ],
                        id="info-modal",
                        size="lg",
                        is_open=False,
                        backdrop="static",
                    )
                ], className="p-5 hero-section")
            ]),

            # ---------------------- CORAL SEGMENTATION ----------------------
            html.Div(id="coral", children=[
                html.H3("Coral Segmentation", className="text-info mb-3"),
                dbc.Card([
                    dbc.CardBody([create_module_tab("Coral")])
                ], className="mb-4")
            ]),

            # ---------------------- POLYP SEGMENTATION ----------------------
            html.Div(id="polyp", children=[
                html.H3("Polyp Segmentation", className="text-info mb-3"),
                dbc.Card([
                    dbc.CardBody([create_module_tab("Polyp")])
                ], className="mb-4")
            ]),

            # ---------------------- HELP SECTION ----------------------
            html.Div(id="help", children=[
                html.H3("Need Help?", className="text-info mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.Textarea(
                            id="chat-log",
                            value="Welcome! Ask me about the platform.\n",
                            style={"width": "100%", "height": "200px"},
                            readOnly=True
                        ),
                        dbc.Input(id="chat-input", placeholder="Type a question...", type="text", className="my-2"),
                        dbc.Button("Send", id="chat-send", color="primary", className="btn btn-primary btn-sm mb-2")
                    ])
                ])
            ])
        ], width=10)
    ])
])


@app.callback(
    Output("tabs", "value"),
    Input("start-now-btn", "n_clicks"),
    prevent_initial_call=True
)
def go_to_polyp(n):
    return "polyp"

def update_tab(tab_value):
    return tab_value
@app.callback(
    Output("info-modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("start-now-btn", "n_clicks")],
    [State("info-modal", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
banner_images = ["/assets/banner1.jpg", "/assets/banner3.jpg", "/assets/banner6.jpg"]

@app.callback(
    Output("banner-img", "src"),
    Output("banner-index", "data"),
    Input("banner-interval", "n_intervals"),
    State("banner-index", "data")
)
def update_banner_image(n_intervals, current_index):
    next_index = (current_index + 1) % len(banner_images)
    return banner_images[next_index], next_index

@app.callback(
    Output({'type': 'tab-content', 'module': MATCH}, 'children'),
    Input({'type': 'child-tabs', 'module': MATCH}, 'active_tab'),
    State({'type': 'original-image', 'module': MATCH}, 'children'),
    State({'type': 'pred-mask', 'module': MATCH}, 'children'),
    State({'type': 'contour-mask', 'module': MATCH}, 'children')
)
def switch_tab(tab, original, pred, contour):
    if 'original' in tab:
        return original
    elif 'mask' in tab:
        return pred
    elif 'contour' in tab:
        return contour
    return ""

@app.callback(
    Output({'type': 'original-image', 'module': MATCH}, 'children'),
    Output({'type': 'pred-mask', 'module': MATCH}, 'children'),
    Output({'type': 'contour-dropdown', 'module': MATCH}, 'options'),
    Output({'type': 'contour-dropdown', 'module': MATCH}, 'value'),
    Output({'type': 'loading-msg', 'module': MATCH}, 'children'),
    Output({'type': 'mask-icons', 'module': MATCH}, 'children'),
    Output({'type': 'mask-metadata', 'module': MATCH}, 'data'),
    Output({'type': 'distance-graph', 'module': MATCH}, 'children'),
    Output({'type': 'ground-truth', 'module': MATCH}, 'children'),
    Input({'type': 'predict-button', 'module': MATCH}, 'n_clicks'),
    State({'type': 'thumb', 'index': ALL, 'module': MATCH}, 'n_clicks'),
    State({'type': 'thumb', 'index': ALL, 'module': MATCH}, 'id'),
    prevent_initial_call=True
)
def predict_and_segment(n_clicks, thumb_clicks, thumb_ids):
    import plotly.graph_objs as go
    if not n_clicks:
        raise PreventUpdate

    image_np = None
    selected = None
    module = None

    max_clicks = -1
    for idx, clicks in enumerate(thumb_clicks):
        if clicks is not None and clicks > max_clicks:
            max_clicks = clicks
            selected = thumb_ids[idx]['index']
            module = thumb_ids[idx]['module']

    if not selected or not module:
        return None, None, [], None, "⚠️ No image selected.", [], [], None, None

    image_path = os.path.join(IMAGE_DIRS[module], selected)
    if not os.path.exists(image_path):
        return None, None, [], None, "⚠️ Image file not found.", [], [], None, None

    image_np = load_raw_image(image_path)

    model = load_model(MODEL_PATHS[module])
    image_pil = Image.fromarray(image_np).resize((256, 256))
    image_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image_pil).unsqueeze(0)

    mask = predict_mask(model, image_tensor)
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours = extract_contours(mask)

    CURRENT[module]['image'] = image_np
    CURRENT[module]['mask'] = mask
    CURRENT[module]['contours'] = contours

    overlay = create_overlay_with_numbers(image_np.copy(), mask, contours, module)
    dropdown, mask_icons, metadata = [], [], []

    for i, c in enumerate(contours):
        area_px = cv2.contourArea(c)
        if module.lower() == "coral" and area_px < 2000:
            continue

        single_mask = np.zeros_like(mask)
        cv2.drawContours(single_mask, [c], -1, 255, thickness=cv2.FILLED)
        pixel_count = np.sum(single_mask > 0)
        pixel_size_mm2 = CURRENT[module].get("pixel_size_mm2", 0.05)
        area_mm2 = pixel_count * pixel_size_mm2

        dropdown.append({"label": f"Mask {i+1} ", "value": i})
        metadata.append({
            "region": i + 1,
            "pixels": int(pixel_count),
            "pixel_size": pixel_size_mm2,
            "area": area_mm2
        })

        icon_overlay = overlay_mask_transparent(image_np, single_mask, alpha=0.6)
        icon_resized = cv2.resize(icon_overlay, (80, 80))
        icon_id = {"type": "mask-icon-img", "module": module, "index": i}
        tooltip_text = (
            f"Mask {i+1} | Pixels: {pixel_count} | "
            f"Size: {pixel_size_mm2:.3f} mm²/pixel | Area: {area_mm2:.2f} mm²"
        )

        mask_icons.append(
            html.Div([
                html.Img(src=encode_np_image(icon_resized), id=icon_id, n_clicks=0),
                dbc.Tooltip(tooltip_text, target=icon_id, placement="top")
            ], style={"display": "inline-block", "margin": "4px"})
        )

    #  Unified scrollable container after 21 icons (3 rows)
    if len(mask_icons) > 21:
        mask_icon_container = html.Div(mask_icons, className="mask-icons-wrapper", style={
            "maxHeight": "270px",  # 3 rows of ~90px each
            "overflowY": "scroll",
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "4px",
            "borderTop": "1px dashed #ccc",
            "paddingTop": "5px"
        })
    else:
        mask_icon_container = html.Div(mask_icons, className="mask-icons-wrapper", style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "4px"
        })

    #  Distance Graph for Polyps Only
    distance_graph = None
    if module.lower() == "polyp" and len(contours) >= 2:
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        distances = []
        for i in range(len(centroids) - 1):
            d = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[i + 1]))
            distances.append((f"Mask {i+1} ↔ {i+2}", d))

        labels = [pair for pair, _ in distances]
        values = [round(val, 2) for _, val in distances]

        distance_graph = dcc.Graph(
            figure={
                "data": [{"x": labels, "y": values, "type": "bar", "name": "Distance"}],
                "layout": {
                    "title": "Distance Between Masks",
                    "xaxis": {"tickangle": -45},
                    "yaxis": {"title": "Pixel Distance"},
                    "margin": {"l": 40, "r": 20, "t": 40, "b": 120}
                }
            }
        )

    #  Area Graph for Coral Ground Truth Only
    ground_truth_mask = None
    if module.lower() == "coral" and metadata:
        ground_truth_mask = dcc.Graph(
            figure=go.Figure(
                data=[go.Bar(
                    x=[f"Region {m['region']}" for m in metadata],
                    y=[m['area'] for m in metadata],
                    marker_color="teal"
                )],
                layout=go.Layout(
                    title="Area of Coral Regions",
                    xaxis={"title": "Region"},
                    yaxis={"title": "Area (mm²)"},
                    height=350
                )
            )
        )

    return (
        html.Div(html.Img(src=encode_np_image(image_np), className="prediction-image-large")),
        html.Div(html.Img(src=encode_np_image(overlay), className="prediction-image-large")),
        dropdown,
        0 if dropdown else None,
        " Prediction complete.",
        mask_icon_container,
        metadata,
        distance_graph,
        ground_truth_mask
    )

@app.callback(
    Output({'type': 'imagej-report', 'module': MATCH}, 'children'),
    Input({'type': 'calc-btn', 'module': MATCH}, 'n_clicks'),
    State({'type': 'scale-canvas', 'module': MATCH}, 'json_data'),
    State({'type': 'contour-dropdown', 'module': MATCH}, 'value'),
    State({'type': 'predict-button', 'module': MATCH}, 'id'),
    prevent_initial_call=True
)
def calculate_metrics(n_clicks, json_data, contour_idx, btn_id):
    module = btn_id['module']

    if not json_data:
        return "❗ Please draw a 2-point line on the image."

    try:
        shapes = json.loads(json_data)["objects"]
        line = next(obj for obj in shapes if obj.get("type") == "line")
        x0, y0 = line["x1"], line["y1"]
        x1, y1 = line["x2"], line["y2"]
    except (KeyError, StopIteration, json.JSONDecodeError):
        return "❗ Please draw a valid 2-point line using the line tool."

    pixel_length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    known_mm = 5.0

    if contour_idx is None:
        return "❗ Please select a mask for area calculation."

    contour = CURRENT[module]['contours'][contour_idx]
    mask = CURRENT[module]['mask']
    image = CURRENT[module]['image']
    height, width = image.shape[:2]

    pixel_size = known_mm / pixel_length
    pixel_area = pixel_size ** 2
    pixel_count = int(cv2.contourArea(contour))
    mask_area = pixel_count * pixel_area
    aspect_ratio = width / height

    return html.Ul([
        html.Li(f" Pixel Distance = {pixel_length:.2f} px"),
        html.Li(f" Known Distance = {known_mm} mm"),
        html.Li(f" Pixel Size = {pixel_size:.4f} mm/px"),
        html.Li(f" Pixel Area = {pixel_area:.4f} mm²/pixel"),
        html.Li(f" Mask Area = {mask_area:.2f} mm²"),
        html.Li(f" Aspect Ratio = {aspect_ratio:.2f}")
    ])
@app.callback(
    Output({'type': 'contour-mask', 'module': MATCH}, 'children'),
    Output({'type': 'mask-base64', 'module': MATCH}, 'data'),
    Input({'type': 'contour-dropdown', 'module': MATCH}, 'value'),
    State({'type': 'predict-button', 'module': MATCH}, 'id'),
    prevent_initial_call=True
)
def update_contour(idx, btn_id):
    module = btn_id['module']
    if idx is None or idx >= len(CURRENT[module]['contours']):
        return " No contour selected.", None

    mask = CURRENT[module]['mask']
    image = CURRENT[module]['image']
    contour = CURRENT[module]['contours'][idx]

    blank = np.zeros_like(mask)
    cv2.drawContours(blank, CURRENT[module]['contours'], idx, color=255, thickness=-1)
    overlay = overlay_mask_transparent(image, blank)
    encoded_overlay = encode_np_image(overlay)

    area = cv2.contourArea(contour)
    pixel_count = np.sum(blank > 0)
    pixel_size_mm2 = 0.01
    area_mm2 = pixel_count * pixel_size_mm2
    height, width = image.shape[:2]
    aspect_ratio = width / height

    details = [
        f"Mask: {idx + 1}",
        f"Pixel Count: {pixel_count}",
        f"Pixel Size: {pixel_size_mm2:.3f} mm²/pixel",
        f"Approx Area: {area_mm2:.2f} mm²",
        f"Aspect Ratio: {aspect_ratio:.2f}"
    ]

    if module.lower() == 'polyp' and idx + 1 < len(CURRENT[module]['contours']):
        cnt2 = CURRENT[module]['contours'][idx + 1]
        M1 = cv2.moments(contour)
        M2 = cv2.moments(cnt2)
        if M1["m00"] > 0 and M2["m00"] > 0:
            cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
            cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
            dist = np.linalg.norm([cx1 - cx2, cy1 - cy2]) * np.sqrt(pixel_size_mm2)
            details.append(f"Distance to Mask {idx+2}: {dist:.2f} mm")

    return html.Div([
        html.Img(src=encoded_overlay, className="prediction-image-large"),
        html.Div(html.Ul([html.Li(d) for d in details]), className="quantification-report")
    ]), encoded_overlay
    
@app.callback(
    Output({'type': 'scale-canvas', 'module': MATCH}, 'image_content'),
    Input({'type': 'mask-base64', 'module': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_canvas_background(encoded):
    triggered_id = ctx.triggered_id
    if not triggered_id or "module" not in triggered_id:
        raise PreventUpdate

    module = triggered_id["module"]
    if module.lower() != "coral":
        raise PreventUpdate

    return encoded
@app.callback(
    Output({'type': 'download-component', 'module': MATCH}, 'data'),
    Input({'type': 'download-button', 'module': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def export_data(n_clicks):
    ctx = dash.callback_context
    module = eval(ctx.triggered[0]['prop_id'].split('.')[0])["module"]
    zip_path = save_mask_and_contours(module)
    return send_file(zip_path)

@app.callback(
    Output("chatbot-container", "style"),
    Input("toggle-chatbot-btn", "n_clicks"),
    State("chatbot-container", "style"),
    prevent_initial_call=True
)
def toggle_chatbot_visibility(n_clicks, current_style):
    if current_style is None or current_style.get("display") == "none":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output("chat-log", "value"),
    Input("chat-send", "n_clicks"),
    State("chat-input", "value"),
    State("chat-log", "value"),
    prevent_initial_call=True
)
def respond_to_input(n_clicks, user_input, history):
    if not user_input:
        raise PreventUpdate
    if history is None:
        history = ""

    user_input_lower = user_input.strip().lower()

    if "what does this platform do" in user_input_lower:
        response = "This platform enables web-based segmentation of coral and polyp images using DeepLabV3++."
    elif "steps of execution" in user_input_lower or "how does it work" in user_input_lower:
        response = (
            "Steps:\n"
            "1. Select a dataset tab.\n"
            "2. Choose an image.\n"
            "3. Run prediction.\n"
            "4. View results and download them."
        )
    elif "purpose" in user_input_lower:
        response = (
            "The tool is designed to help researchers visualize and quantify segmented coral and polyp regions from scientific images."
        )
    else:
        response = "Thanks for your question. I'll get back with more details soon."

    new_history = history + f"\nYou: {user_input}\nAssistant: {response}"
    return new_history

if __name__ == '__main__':
    app.run_server(debug=False)
