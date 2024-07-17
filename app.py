import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'modified_fasterrcnn_resnet50_fpn.pth'

def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

num_classes = 2  
model = create_model(num_classes)
model.load_state_dict(torch.load(app.config['MODEL_PATH']))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def detect_wheat(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device)
    
    with torch.no_grad():
        output = model([image_tensor])
    
    im = image.copy()
    draw = ImageDraw.Draw(im)
    
    for idx in range(len(output[0]['boxes'])):
        box = output[0]['boxes'][idx].cpu().numpy()
        score = output[0]['scores'][idx].cpu().numpy()
        if score > 0.5: 
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
    
    return im

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_image = detect_wheat(filepath)
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_image.save(result_path)
            return render_template('index.html', filename=result_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    else:
        os.abort(404)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    app.run(debug=True,port=8000)
