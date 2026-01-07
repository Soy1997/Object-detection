from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTION_FOLDER'] = 'predictions'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)

model = "YOLOv7"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_data():
    weights = f"weights/{model.lower()}.weights"
    cfg = f"cfg/{model.lower()}.cfg"
    net = cv2.dnn.readNet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    classes = []
    with open("data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] 
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), 
                                mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def process_image(image_path):
    net, classes, colors, output_layers = load_data()
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    _, outputs = detect_objects(img, net, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confs[i]
            color = colors[class_ids[i]]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                f"{label} {confidence:.2f}", 
                font, 
                0.6, 
                1
            )
            cv2.rectangle(
                img, 
                (x, y - text_height - baseline), 
                (x + text_width, y), 
                color, 
                cv2.FILLED
            )
            
            cv2.putText(
                img, 
                f"{label} {confidence:.2f}", 
                (x, y - baseline), 
                font, 
                0.6, 
                (0, 0, 0),
                1
            )
    
    filename = secure_filename(os.path.basename(image_path))
    output_path = os.path.join(app.config['PREDICTION_FOLDER'], filename)
    cv2.imwrite(output_path, img)
    
    return filename
def gen_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    net, classes, colors, output_layers = load_data()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        _, outputs = detect_objects(frame, net, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confs[i]
                color = colors[class_ids[i]]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    f"{label} {confidence:.2f}", 
                    font, 
                    0.6, 
                    1
                )
                cv2.rectangle(
                    frame, 
                    (x, y - text_height - baseline), 
                    (x + text_width, y), 
                    color, 
                    cv2.FILLED
                )
                
                cv2.putText(
                    frame, 
                    f"{label} {confidence:.2f}", 
                    (x, y - baseline), 
                    font, 
                    0.6, 
                    (0, 0, 0),
                    1
                )
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam_feed', methods=['POST'])
def webcam():
    global model
    model = str(request.form['model'])
    if model not in ["YOLOv7", "YOLOv8", "Faster R-CNN"]:
        return "Invalid model selected", 400
    return render_template('webcam.html')

@app.route('/image_detection', methods=['POST'])
def handle_image_detection():
    global model
    model = str(request.form.get('model', 'YOLOv7'))
    
    if 'image-upload' not in request.files:
        return "No file part", 400
        
    file = request.files['image-upload']
    if file.filename == '':
        return "No selected file", 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        processed_filename = process_image(upload_path)
        return redirect(url_for('show_prediction', filename=processed_filename))
    
    return "Invalid file type", 400

@app.route('/predictions/<filename>')
def show_prediction(filename):
    return render_template('result.html', filename=filename)


@app.route('/get_prediction/<filename>')
def get_prediction(filename):
    return send_from_directory(app.config['PREDICTION_FOLDER'], filename)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)