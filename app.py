# app.py
import os
import uuid
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'coralguard-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ======================== DISEASE DESCRIPTIONS ========================
DISEASE_INFO = {
    'Band disease': {
        'description': 'Band disease is characterized by a distinct band of tissue '
                       'loss that progresses across the coral colony. It is caused by '
                       'cyanobacteria and other microbial pathogens that form a dark '
                       'or light band at the interface between healthy tissue and '
                       'exposed skeleton.',
        'severity': 'High',
        'treatment': 'Remove infected portions, improve water quality, and reduce '
                     'nutrient runoff. Marine epoxy can be applied to create barriers.',
        'color': '#e74c3c',
        'icon': '🔴'
    },
    'Bleached disease': {
        'description': 'Coral bleaching occurs when corals expel their symbiotic '
                       'zooxanthellae algae due to stress, primarily from elevated '
                       'water temperatures. The coral turns white as the algae '
                       'provide most of the coral\'s color.',
        'severity': 'Critical',
        'treatment': 'Reduce thermal stress, minimize local stressors such as '
                     'pollution and overfishing. Recovery is possible if conditions '
                     'improve within weeks.',
        'color': '#f39c12',
        'icon': '⚪'
    },
    'Dead Coral': {
        'description': 'Dead coral has completely lost all living tissue, leaving '
                       'behind only the calcium carbonate skeleton. The skeleton '
                       'often becomes covered with algae over time. Dead coral '
                       'no longer contributes to reef growth.',
        'severity': 'Terminal',
        'treatment': 'Coral restoration through transplantation of healthy coral '
                     'fragments. Artificial reef structures can help provide new '
                     'substrate for coral settlement.',
        'color': '#7f8c8d',
        'icon': '💀'
    },
    'Healthy Coral': {
        'description': 'Healthy coral exhibits vibrant coloration, active polyp '
                       'extension, and normal growth patterns. The symbiotic '
                       'zooxanthellae algae are present, providing energy through '
                       'photosynthesis and giving the coral its characteristic colors.',
        'severity': 'None',
        'treatment': 'Continue monitoring and maintain good water quality. '
                     'Protect from physical damage and environmental stressors.',
        'color': '#27ae60',
        'icon': '✅'
    },
    'White Pox Disease': {
        'description': 'White Pox Disease (also known as acroporid serratiosis) '
                       'is caused by the bacterium Serratia marcescens. It creates '
                       'irregular white patches or lesions on the coral surface '
                       'where tissue has been lost, exposing the skeleton.',
        'severity': 'High',
        'treatment': 'Reduce sewage and agricultural runoff. Antibiotic treatments '
                     'in controlled settings. Quarantine affected colonies to '
                     'prevent spread.',
        'color': '#9b59b6',
        'icon': '🟣'
    }
}

# ======================== DATABASE MODELS ========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    detections = db.relationship('Detection', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    detection_type = db.Column(db.String(20), nullable=False)  # 'coral' or 'fish'
    original_image = db.Column(db.String(256), nullable=False)
    result_image = db.Column(db.String(256), nullable=False)
    results_json = db.Column(db.Text, nullable=False)
    confidence_avg = db.Column(db.Float, default=0.0)
    total_detections = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ======================== LOAD YOLO MODELS ========================
# Load models (make sure model files exist)
coral_model = None
fish_model = None

def load_models():
    global coral_model, fish_model
    try:
        coral_model_path = 'models/coral_model.pt'
        if os.path.exists(coral_model_path):
            coral_model = YOLO(coral_model_path)
            print("✅ Coral detection model loaded successfully!")
        else:
            print(f"⚠️ Coral model not found at {coral_model_path}")
            print("   Using YOLOv8n as placeholder. Replace with your trained model.")
            coral_model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"❌ Error loading coral model: {e}")

    try:
        fish_model_path = 'models/fish_model.pt'
        if os.path.exists(fish_model_path):
            fish_model = YOLO(fish_model_path)
            print("✅ Fish detection model loaded successfully!")
        else:
            print(f"⚠️ Fish model not found at {fish_model_path}")
            print("   Using YOLOv8n as placeholder. Replace with your trained model.")
            fish_model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"❌ Error loading fish model: {e}")


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ======================== ROUTES ========================

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        errors = []
        if not full_name or len(full_name) < 2:
            errors.append('Full name must be at least 2 characters.')
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if not email:
            errors.append('Email is required.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already exists.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        user = User(
            full_name=full_name,
            username=username,
            email=email
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)

        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()

        if user and user.check_password(password):
            login_user(user, remember=bool(remember))
            flash(f'Welcome back, {user.full_name}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))


@app.route('/dashboard')
@login_required
def dashboard():
    total_detections = Detection.query.filter_by(user_id=current_user.id).count()
    coral_detections = Detection.query.filter_by(
        user_id=current_user.id, detection_type='coral'
    ).count()
    fish_detections = Detection.query.filter_by(
        user_id=current_user.id, detection_type='fish'
    ).count()
    recent = Detection.query.filter_by(
        user_id=current_user.id
    ).order_by(Detection.created_at.desc()).limit(5).all()

    return render_template('dashboard.html',
                           total_detections=total_detections,
                           coral_detections=coral_detections,
                           fish_detections=fish_detections,
                           recent_detections=recent)


@app.route('/detect/coral', methods=['GET', 'POST'])
@login_required
def detect_coral():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save original image
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run YOLO detection
            try:
                results = coral_model.predict(
                    source=filepath,
                    conf=0.25,
                    save=False,
                    verbose=False
                )

                result = results[0]
                detections_list = []
                class_names = ['Band disease', 'Bleached disease',
                               'Dead Coral', 'Healthy Coral', 'White Pox Disease']

                # Draw bounding boxes on image
                img = cv2.imread(filepath)
                img_result = img.copy()

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # Get class name
                    if cls_id < len(class_names):
                        cls_name = class_names[cls_id]
                    else:
                        cls_name = result.names.get(cls_id, f'Class_{cls_id}')

                    # Get color for this disease
                    info = DISEASE_INFO.get(cls_name, {})
                    hex_color = info.get('color', '#3498db')
                    # Convert hex to BGR
                    hex_color = hex_color.lstrip('#')
                    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                    bgr = (rgb[2], rgb[1], rgb[0])

                    # Draw box
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), bgr, 3)

                    # Label
                    label = f"{cls_name}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        img_result,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        bgr, -1
                    )
                    cv2.putText(
                        img_result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

                    detection_data = {
                        'class': cls_name,
                        'confidence': round(conf * 100, 2),
                        'bbox': [x1, y1, x2, y2],
                        'info': DISEASE_INFO.get(cls_name, {})
                    }
                    detections_list.append(detection_data)

                # Save result image
                result_filename = f"result_{filename}"
                result_filepath = os.path.join(
                    app.config['RESULT_FOLDER'], result_filename
                )
                cv2.imwrite(result_filepath, img_result)

                # Calculate avg confidence
                avg_conf = (
                    sum(d['confidence'] for d in detections_list) / len(detections_list)
                    if detections_list else 0
                )

                # Save to database
                detection_record = Detection(
                    user_id=current_user.id,
                    detection_type='coral',
                    original_image=filename,
                    result_image=result_filename,
                    results_json=json.dumps(detections_list),
                    confidence_avg=round(avg_conf, 2),
                    total_detections=len(detections_list)
                )
                db.session.add(detection_record)
                db.session.commit()

                return render_template('result.html',
                                       detection=detection_record,
                                       detections=detections_list,
                                       detection_type='coral',
                                       disease_info=DISEASE_INFO)

            except Exception as e:
                flash(f'Error during detection: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed: png, jpg, jpeg, webp, bmp', 'danger')
            return redirect(request.url)

    return render_template('coral_detect.html')


@app.route('/detect/fish', methods=['GET', 'POST'])
@login_required
def detect_fish():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                results = fish_model.predict(
                    source=filepath,
                    conf=0.25,
                    save=False,
                    verbose=False
                )

                result = results[0]
                detections_list = []
                fish_count = 0

                img = cv2.imread(filepath)
                img_result = img.copy()

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names.get(cls_id, f'Fish_{cls_id}')

                    fish_count += 1

                    # Draw box in ocean blue
                    color = (255, 165, 0)  # Orange in BGR
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 3)

                    label = f"Fish #{fish_count}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        img_result,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color, -1
                    )
                    cv2.putText(
                        img_result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                    detection_data = {
                        'class': cls_name,
                        'fish_number': fish_count,
                        'confidence': round(conf * 100, 2),
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections_list.append(detection_data)

                # Add fish count overlay
                count_label = f"Total Fish Count: {fish_count}"
                cv2.rectangle(img_result, (10, 10), (350, 60), (0, 0, 0), -1)
                cv2.putText(
                    img_result, count_label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                )

                result_filename = f"result_{filename}"
                result_filepath = os.path.join(
                    app.config['RESULT_FOLDER'], result_filename
                )
                cv2.imwrite(result_filepath, img_result)

                avg_conf = (
                    sum(d['confidence'] for d in detections_list) / len(detections_list)
                    if detections_list else 0
                )

                # Store fish count in results
                results_data = {
                    'fish_count': fish_count,
                    'detections': detections_list
                }

                detection_record = Detection(
                    user_id=current_user.id,
                    detection_type='fish',
                    original_image=filename,
                    result_image=result_filename,
                    results_json=json.dumps(results_data),
                    confidence_avg=round(avg_conf, 2),
                    total_detections=fish_count
                )
                db.session.add(detection_record)
                db.session.commit()

                return render_template('result.html',
                                       detection=detection_record,
                                       detections=detections_list,
                                       fish_count=fish_count,
                                       detection_type='fish',
                                       disease_info=DISEASE_INFO)

            except Exception as e:
                flash(f'Error during detection: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type.', 'danger')
            return redirect(request.url)

    return render_template('fish_detect.html')


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    filter_type = request.args.get('type', 'all')

    query = Detection.query.filter_by(user_id=current_user.id)

    if filter_type == 'coral':
        query = query.filter_by(detection_type='coral')
    elif filter_type == 'fish':
        query = query.filter_by(detection_type='fish')

    detections = query.order_by(
        Detection.created_at.desc()
    ).paginate(page=page, per_page=9, error_out=False)

    return render_template('history.html',
                           detections=detections,
                           filter_type=filter_type)


@app.route('/history/<int:detection_id>')
@login_required
def view_detection(detection_id):
    detection = Detection.query.filter_by(
        id=detection_id, user_id=current_user.id
    ).first_or_404()

    results_data = json.loads(detection.results_json)

    if detection.detection_type == 'fish':
        detections_list = results_data.get('detections', results_data)
        fish_count = results_data.get('fish_count', len(detections_list))
        return render_template('result.html',
                               detection=detection,
                               detections=detections_list,
                               fish_count=fish_count,
                               detection_type='fish',
                               disease_info=DISEASE_INFO)
    else:
        return render_template('result.html',
                               detection=detection,
                               detections=results_data,
                               detection_type='coral',
                               disease_info=DISEASE_INFO)


@app.route('/history/delete/<int:detection_id>', methods=['POST'])
@login_required
def delete_detection(detection_id):
    detection = Detection.query.filter_by(
        id=detection_id, user_id=current_user.id
    ).first_or_404()

    # Delete files
    try:
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], detection.original_image)
        result_path = os.path.join(app.config['RESULT_FOLDER'], detection.result_image)
        if os.path.exists(orig_path):
            os.remove(orig_path)
        if os.path.exists(result_path):
            os.remove(result_path)
    except Exception:
        pass

    db.session.delete(detection)
    db.session.commit()
    flash('Detection record deleted.', 'success')
    return redirect(url_for('history'))


# ======================== INITIALIZE ========================
with app.app_context():
    db.create_all()
    load_models()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)