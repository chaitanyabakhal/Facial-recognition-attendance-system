from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import datetime
import base64
import json
import pickle
import numpy as np
from deepface import DeepFace
from database import get_connection, init_db

# ==========================
# FLASK INITIALIZATION
# ==========================
app = Flask(__name__)
app.secret_key = 'facial_recognition_attendance_system'

# Allow large uploads (for data URLs)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# Ensure folders exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/models", exist_ok=True)
os.makedirs("static/temp", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Init DB
init_db()


# ==========================
# HELPERS
# ==========================
def dict_factory(cursor, row):
    """Return rows as dicts for template access."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


# ==========================
# TRAIN: save only embeddings
# ==========================
def train_face_model(photos_folder, roll_number):
    """Compute and save embeddings for all student photos (vectors only)."""
    models_dir = os.path.join('static', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{roll_number}_model.pkl")

    photo_paths = [
        os.path.join(photos_folder, f)
        for f in os.listdir(photos_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    embeddings = []
    for photo_path in photo_paths:
        try:
            reps = DeepFace.represent(
                img_path=photo_path,
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend="opencv",
                align=True,
                normalization="base",
            )
            vec = reps[0]["embedding"] if isinstance(reps, list) else reps["embedding"]
            embeddings.append(np.array(vec, dtype="float32"))
        except Exception as e:
            print(f"⚠️ Error processing {photo_path}: {str(e)}")

    with open(model_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return model_path


# ==========================
# ROUTES
# ==========================
@app.route('/')
def index():
    return render_template('index.html')


# --------------------------
# REGISTER STUDENT
# --------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')
        department = request.form.get('department')
        year = request.form.get('year')
        photos_data = request.form.get('photos_data')

        if not name or not roll_number or not department or not year or not photos_data:
            flash('⚠️ All fields are required!', 'error')
            return redirect(url_for('register'))

        try:
            student_dir = os.path.join('static', 'uploads', roll_number)
            os.makedirs(student_dir, exist_ok=True)

            photos_list = json.loads(photos_data)

            conn = get_connection()
            c = conn.cursor()

            # Insert student info
            c.execute('''
                INSERT INTO students (name, roll_number, department, year)
                VALUES (?, ?, ?, ?)
            ''', (name, roll_number, department, year))
            student_id = c.lastrowid

            # Save captured photos
            for i, photo_data in enumerate(photos_list):
                photo_data = photo_data.split(',')[1]
                photo_binary = base64.b64decode(photo_data)
                photo_filename = f"{roll_number}_{i + 1}.jpg"
                photo_path = os.path.join(student_dir, photo_filename)

                with open(photo_path, 'wb') as f:
                    f.write(photo_binary)

                c.execute('''
                    INSERT INTO student_photos (student_id, photo_path)
                    VALUES (?, ?)
                ''', (student_id, photo_path))

            # Train model (save embeddings)
            model_path = train_face_model(student_dir, roll_number)

            c.execute('UPDATE students SET face_model_path = ? WHERE id = ?', (model_path, student_id))
            conn.commit()
            conn.close()

            flash('✅ Student registered successfully!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'❌ Error: {str(e)}', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')


# --------------------------
# TAKE ATTENDANCE (page)
# --------------------------
@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')


# --------------------------
# PROCESS ATTENDANCE (API)
# --------------------------
@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    def cosine_distance(a, b):
        eps = 1e-10
        return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image provided'})

        # decode data URL -> bytes
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)

        # save temp
        temp_path = os.path.join('static', 'temp', 'captured.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        # embed captured frame
        reps = DeepFace.represent(
            img_path=temp_path,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
            normalization="base",
        )
        probe_vec = reps[0]["embedding"] if isinstance(reps, list) else reps["embedding"]
        probe_vec = np.array(probe_vec, dtype="float32")

        # fetch registered students
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id, name, roll_number, face_model_path FROM students WHERE face_model_path IS NOT NULL')
        students = c.fetchall()
        if not students:
            conn.close()
            return jsonify({'success': False, 'message': 'No registered students found'})

        THRESH = 0.35  # tune 0.30–0.45 for VGG-Face + cosine
        best_match = None  # (student_tuple, best_dist)

        for student_id, name, roll_number, model_path in students:
            try:
                with open(model_path, 'rb') as f:
                    gallery_vectors = pickle.load(f)  # list of numpy vectors

                if not gallery_vectors:
                    continue

                dists = [cosine_distance(probe_vec, np.array(vec, dtype="float32")) for vec in gallery_vectors]
                min_dist = float(np.min(dists))

                print(f"🔍 {roll_number} min_cosine_dist={min_dist:.4f}")

                if min_dist <= THRESH:
                    if best_match is None or min_dist < best_match[1]:
                        best_match = ((student_id, name, roll_number), min_dist)
            except Exception as e:
                print(f"⚠️ Error matching {roll_number}: {str(e)}")
                continue

        if best_match is None:
            conn.close()
            return jsonify({'success': False, 'message': 'No match found'})

        student_id, name, roll_number = best_match[0]
        today = datetime.datetime.now().strftime('%Y-%m-%d')

        # Already marked today?
        c.execute('SELECT id FROM attendance WHERE student_id = ? AND date = ?', (student_id, today))
        if c.fetchone():
            conn.close()
            return jsonify({
                'success': True,
                'message': f'Attendance already marked for {name} ({roll_number})',
                'already_marked': True
            })

        now = datetime.datetime.now()
        c.execute('INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)',
                  (student_id, today, now.strftime('%H:%M:%S')))
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': f'✅ Attendance marked for {name} ({roll_number})',
            'student': {'id': student_id, 'name': name, 'roll_number': roll_number}
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


# --------------------------
# VIEW ATTENDANCE
# --------------------------
@app.route('/view_attendance')
def view_attendance():
    conn = get_connection()
    conn.row_factory = dict_factory
    c = conn.cursor()
    c.execute('''
        SELECT a.id, a.date, a.time, s.name, s.roll_number, s.department, s.year
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.date DESC, a.time DESC
    ''')
    attendance_records = c.fetchall()
    conn.close()
    return render_template('view_attendance.html', attendance_records=attendance_records)


# --------------------------
# VIEW STUDENTS
# --------------------------
@app.route('/students')
def students():
    conn = get_connection()
    conn.row_factory = dict_factory
    c = conn.cursor()
    c.execute('''
        SELECT s.*,
               (SELECT photo_path 
                  FROM student_photos sp 
                 WHERE sp.student_id = s.id 
                 ORDER BY sp.id 
                 LIMIT 1) AS image_path
        FROM students s
        ORDER BY s.name
    ''')
    students_list = c.fetchall()
    conn.close()
    return render_template('students.html', students=students_list)


# ==========================
# MAIN
# ==========================
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
