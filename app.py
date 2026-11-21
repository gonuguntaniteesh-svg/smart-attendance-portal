import os
import sys
import base64
import json
import datetime
import sqlite3
import cv2
import numpy as np
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_file
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# Ensure deployment directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")
sys.path.insert(0, DEPLOYMENT_DIR)

from database import init_db, get_connection
from face_engine import FaceRecognitionEngine
from attendance_manager import AttendanceManager

app = Flask(__name__)
app.secret_key = "smart-attendance-portal-secret-key-vit-svm"
app.config["UPLOAD_FOLDER"] = os.path.join(DEPLOYMENT_DIR, "static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Pre-initialize DB and Model Engine
init_db()
engine = FaceRecognitionEngine.get_instance()
attendance_mgr = AttendanceManager.get_instance()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Please sign in as administrator to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ==========================================
# 1. AUTHENTICATION (ADMIN LOGIN / LOGOUT)
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username = ?", (username,))
        admin = cursor.fetchone()
        conn.close()

        if admin and check_password_hash(admin["password_hash"], password):
            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]
            session["admin_name"] = admin["full_name"]
            session["admin_role"] = admin["role"]
            flash(f"Welcome back, {admin['full_name']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid administrator credentials. Please check your username and password.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("login"))

# ==========================================
# 2. DASHBOARD
# ==========================================

@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_connection()
    cursor = conn.cursor()

    # Stat 1: Total Students
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    # Stat 2: Enrolled Students
    cursor.execute("SELECT COUNT(*) FROM students WHERE is_enrolled = 1")
    enrolled_students = cursor.fetchone()[0]

    # Stat 3: Total Faculty
    cursor.execute("SELECT COUNT(*) FROM faculty")
    total_faculty = cursor.fetchone()[0]

    # Stat 4: Today's Attendance Count
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date = ?", (today_str,))
    today_attendance_count = cursor.fetchone()[0]

    # Department breakdown
    cursor.execute("SELECT department, COUNT(*) as count FROM students GROUP BY department")
    dept_rows = cursor.fetchall()
    dept_labels = [r["department"] for r in dept_rows]
    dept_counts = [r["count"] for r in dept_rows]

    # Recent Attendance Logs for Today
    cursor.execute("""
        SELECT * FROM attendance 
        WHERE attendance_date = ? 
        ORDER BY id DESC LIMIT 10
    """, (today_str,))
    recent_logs = [dict(r) for r in cursor.fetchall()]

    conn.close()

    stats = {
        "total_students": total_students,
        "enrolled_students": enrolled_students,
        "total_faculty": total_faculty,
        "today_attendance_count": today_attendance_count
    }

    return render_template(
        "dashboard.html",
        stats=stats,
        dept_labels=dept_labels,
        dept_counts=dept_counts,
        recent_logs=recent_logs,
        today_date=today_str
    )

# ==========================================
# 3. STUDENT MANAGEMENT (CRUD)
# ==========================================

@app.route("/students")
@login_required
def list_students():
    search_query = request.args.get("search", "").strip()
    dept_filter = request.args.get("department", "").strip()
    year_filter = request.args.get("academic_year", "").strip()

    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM students WHERE 1=1"
    params = []

    if search_query:
        query += " AND (id LIKE ? OR full_name LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    if dept_filter:
        query += " AND department = ?"
        params.append(dept_filter)
    if year_filter:
        query += " AND academic_year = ?"
        params.append(year_filter)

    query += " ORDER BY id ASC"
    cursor.execute(query, params)
    students = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM departments ORDER BY code ASC")
    departments = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM academic_years ORDER BY id ASC")
    academic_years = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return render_template(
        "students.html",
        students=students,
        departments=departments,
        academic_years=academic_years,
        search_query=search_query,
        dept_filter=dept_filter,
        year_filter=year_filter
    )

@app.route("/students/add", methods=["GET", "POST"])
@login_required
def add_student():
    conn = get_connection()
    cursor = conn.cursor()

    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip().upper()
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()
        academic_year = request.form.get("academic_year", "").strip()
        section = request.form.get("section", "").strip()
        email = request.form.get("email", "").strip()
        parent_email = request.form.get("parent_email", "").strip()
        action = request.form.get("action", "save")

        # Check unique ID
        cursor.execute("SELECT id FROM students WHERE id = ?", (student_id,))
        if cursor.fetchone():
            conn.close()
            flash(f"Student ID '{student_id}' is already registered.", "error")
            return redirect(url_for("add_student"))

        photo_path = None
        if "photo" in request.files:
            file = request.files["photo"]
            if file and file.filename != "":
                filename = secure_filename(f"{student_id}_{file.filename}")
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                photo_path = f"uploads/{filename}"

        cursor.execute("""
            INSERT INTO students 
            (id, full_name, department, academic_year, section, email, parent_email, photo_path, is_enrolled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (student_id, full_name, department, academic_year, section, email, parent_email, photo_path))
        conn.commit()
        conn.close()

        flash(f"Student '{full_name}' ({student_id}) registered successfully!", "success")

        if action == "save_and_enroll":
            return redirect(url_for("enroll_face", student_id=student_id))
        return redirect(url_for("list_students"))

    cursor.execute("SELECT * FROM departments ORDER BY code ASC")
    departments = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM academic_years ORDER BY id ASC")
    academic_years = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM sections ORDER BY id ASC")
    sections = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        "student_form.html",
        student=None,
        departments=departments,
        academic_years=academic_years,
        sections=sections
    )

@app.route("/students/edit/<student_id>", methods=["GET", "POST"])
@login_required
def edit_student(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()
        academic_year = request.form.get("academic_year", "").strip()
        section = request.form.get("section", "").strip()
        email = request.form.get("email", "").strip()
        parent_email = request.form.get("parent_email", "").strip()

        cursor.execute("""
            UPDATE students 
            SET full_name = ?, department = ?, academic_year = ?, section = ?, email = ?, parent_email = ?
            WHERE id = ?
        """, (full_name, department, academic_year, section, email, parent_email, student_id))
        conn.commit()
        conn.close()

        flash(f"Updated record for '{full_name}' successfully.", "success")
        return redirect(url_for("list_students"))

    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    if not student:
        conn.close()
        flash("Student not found.", "error")
        return redirect(url_for("list_students"))

    cursor.execute("SELECT * FROM departments ORDER BY code ASC")
    departments = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM academic_years ORDER BY id ASC")
    academic_years = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM sections ORDER BY id ASC")
    sections = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        "student_form.html",
        student=dict(student),
        departments=departments,
        academic_years=academic_years,
        sections=sections
    )

@app.route("/students/delete/<student_id>")
@login_required
def delete_student(student_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT full_name FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()

    if student:
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.commit()
        flash(f"Student '{student['full_name']}' ({student_id}) removed from system.", "info")
    else:
        flash("Student not found.", "error")

    conn.close()
    return redirect(url_for("list_students"))

# ==========================================
# 4. BIOMETRIC FACE ENROLLMENT
# ==========================================

@app.route("/students/enroll/<student_id>")
@login_required
def enroll_face(student_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()

    if not student:
        flash("Student not found for enrollment.", "error")
        return redirect(url_for("list_students"))

    return render_template("enroll_face.html", student=dict(student))

@app.route("/api/enroll_faces/<student_id>", methods=["POST"])
@login_required
def api_enroll_faces(student_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()

    if not student:
        conn.close()
        return jsonify({"success": False, "message": "Student not found"}), 404

    data = request.get_json()
    if not data or "images" not in data or len(data["images"]) == 0:
        conn.close()
        return jsonify({"success": False, "message": "No images provided"}), 400

    images_b64 = data["images"]
    face_crops = []

    for b64 in images_b64:
        try:
            if "," in b64:
                b64 = b64.split(",")[1]
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # Detect face
                faces = engine.detect_faces(img)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    # Add 10% padding
                    pad_w = int(w * 0.1)
                    pad_h = int(h * 0.1)
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(img.shape[1], x + w + pad_w)
                    y2 = min(img.shape[0], y + h + pad_h)
                    crop = img[y1:y2, x1:x2]
                    face_crops.append(crop)
                else:
                    face_crops.append(img)
        except Exception as e:
            print("Image decode error during enrollment:", e)

    if len(face_crops) == 0:
        conn.close()
        return jsonify({"success": False, "message": "No valid faces could be extracted"}), 400

    # Enroll in Face Recognition Engine
    success, msg = engine.enroll_student_faces(student["full_name"], face_crops)

    if success:
        cursor.execute("""
            UPDATE students 
            SET is_enrolled = 1, enrolled_images_count = ? 
            WHERE id = ?
        """, (len(face_crops), student_id))
        conn.commit()

    conn.close()
    return jsonify({"success": success, "message": msg})

# ==========================================
# 5. LIVE AUTOMATIC ATTENDANCE
# ==========================================

@app.route("/attendance/live")
@login_required
def live_attendance():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects ORDER BY code ASC")
    subjects = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return render_template("live_attendance.html", subjects=subjects)

@app.route("/api/process_frame", methods=["POST"])
@login_required
def api_process_frame():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"faces": []}), 400

    subject = data.get("subject", "Deep Learning & Vision")
    session_name = data.get("session_name", "Regular Lecture")
    threshold = float(data.get("threshold", 0.65))

    frame_b64 = data["frame"]
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",")[1]

    img_bytes = base64.b64decode(frame_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"faces": []})

    detected_faces = engine.detect_faces(frame)
    results = []

    for (x, y, w, h) in detected_faces:
        face_crop = frame[y:y+h, x:x+w]
        pred = engine.predict_identity(face_crop, threshold=threshold)

        face_entry = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "is_known": pred["is_known"],
            "name": pred["identity"].replace("_", " "),
            "confidence": pred["confidence"],
            "attendance_result": None
        }

        # If known face, automatically log attendance with duplicate prevention
        if pred["is_known"]:
            success, msg, rec = attendance_mgr.mark_attendance(
                identity_name=pred["identity"],
                confidence=pred["confidence"],
                subject=subject,
                session_name=session_name
            )
            face_entry["attendance_result"] = {
                "success": success,
                "message": msg,
                "record": rec
            }
            if rec:
                face_entry["student_id"] = rec["student_id"]

        results.append(face_entry)

    return jsonify({"faces": results})

# ==========================================
# 6. ATTENDANCE RECORDS & EXCEL EXPORT
# ==========================================

@app.route("/attendance/records")
@login_required
def attendance_records():
    date_filter = request.args.get("date", "").strip()
    dept_filter = request.args.get("department", "").strip()
    subject_filter = request.args.get("subject", "").strip()

    records = attendance_mgr.get_attendance_records(
        date_filter=date_filter or None,
        department_filter=dept_filter or None,
        subject_filter=subject_filter or None
    )

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM departments ORDER BY code ASC")
    departments = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM subjects ORDER BY code ASC")
    subjects = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        "attendance_records.html",
        records=records,
        departments=departments,
        subjects=subjects,
        date_filter=date_filter,
        dept_filter=dept_filter,
        subject_filter=subject_filter
    )

@app.route("/attendance/export")
@login_required
def export_excel():
    date_filter = request.args.get("date", "").strip()
    dept_filter = request.args.get("department", "").strip()
    subject_filter = request.args.get("subject", "").strip()

    filepath = attendance_mgr.export_attendance_excel(
        date_filter=date_filter or None,
        department_filter=dept_filter or None,
        subject_filter=subject_filter or None
    )

    return send_file(filepath, as_attachment=True)

# ==========================================
# 7. ACADEMIC STRUCTURE SETUP
# ==========================================

@app.route("/academic")
@login_required
def academic_setup():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM departments ORDER BY code ASC")
    departments = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM subjects ORDER BY code ASC")
    subjects = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM faculty ORDER BY id ASC")
    faculty = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        "academic_setup.html",
        departments=departments,
        subjects=subjects,
        faculty=faculty
    )

@app.route("/academic/department/add", methods=["POST"])
@login_required
def add_department():
    code = request.form.get("code", "").strip().upper()
    name = request.form.get("name", "").strip()
    if code and name:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO departments (code, name) VALUES (?, ?)", (code, name))
            conn.commit()
            flash(f"Department '{code}' added successfully.", "success")
        except sqlite3.IntegrityError:
            flash(f"Department code '{code}' already exists.", "error")
        conn.close()
    return redirect(url_for("academic_setup"))

@app.route("/academic/subject/add", methods=["POST"])
@login_required
def add_subject():
    code = request.form.get("code", "").strip().upper()
    name = request.form.get("name", "").strip()
    dept = request.form.get("department", "").strip()
    if code and name and dept:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO subjects (code, name, department) VALUES (?, ?, ?)", (code, name, dept))
            conn.commit()
            flash(f"Subject '{name}' ({code}) added successfully.", "success")
        except sqlite3.IntegrityError:
            flash(f"Subject code '{code}' already exists.", "error")
        conn.close()
    return redirect(url_for("academic_setup"))

@app.route("/academic/faculty/add", methods=["POST"])
@login_required
def add_faculty():
    faculty_id = request.form.get("faculty_id", "").strip().upper()
    name = request.form.get("name", "").strip()
    dept = request.form.get("department", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()

    if faculty_id and name and dept:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO faculty (faculty_id, name, department, email, phone) 
                VALUES (?, ?, ?, ?, ?)
            """, (faculty_id, name, dept, email, phone))
            conn.commit()
            flash(f"Faculty member '{name}' added successfully.", "success")
        except sqlite3.IntegrityError:
            flash(f"Faculty ID '{faculty_id}' already exists.", "error")
        conn.close()
    return redirect(url_for("academic_setup"))

# ==========================================
# 8. RESEARCH & BENCHMARKS VIEW
# ==========================================

@app.route("/research")
@login_required
def research_view():
    results_table = [
        {"extractor": "ResNet50", "dim": "2048", "classifier": "SVM", "accuracy": "62.50", "precision": "68.45", "recall": "62.50", "f1": "61.57"},
        {"extractor": "ResNet50", "dim": "2048", "classifier": "XGBoost", "accuracy": "65.00", "precision": "60.38", "recall": "65.00", "f1": "61.32"},
        {"extractor": "ResNet50", "dim": "2048", "classifier": "AdaBoost", "accuracy": "40.00", "precision": "41.05", "recall": "40.00", "f1": "35.86"},
        {"extractor": "ResNet50", "dim": "2048", "classifier": "Gradient Boosting", "accuracy": "50.00", "precision": "41.50", "recall": "50.00", "f1": "44.91"},
        {"extractor": "MobileNetV2", "dim": "1280", "classifier": "SVM", "accuracy": "50.00", "precision": "49.33", "recall": "50.00", "f1": "46.90"},
        {"extractor": "MobileNetV2", "dim": "1280", "classifier": "XGBoost", "accuracy": "42.50", "precision": "42.00", "recall": "42.50", "f1": "40.57"},
        {"extractor": "MobileNetV2", "dim": "1280", "classifier": "AdaBoost", "accuracy": "30.00", "precision": "37.17", "recall": "30.00", "f1": "30.00"},
        {"extractor": "MobileNetV2", "dim": "1280", "classifier": "Gradient Boosting", "accuracy": "45.00", "precision": "45.25", "recall": "45.00", "f1": "42.59"},
        {"extractor": "InceptionV3", "dim": "2048", "classifier": "SVM", "accuracy": "72.50", "precision": "79.33", "recall": "72.50", "f1": "71.29"},
        {"extractor": "InceptionV3", "dim": "2048", "classifier": "XGBoost", "accuracy": "60.00", "precision": "64.33", "recall": "60.00", "f1": "59.21"},
        {"extractor": "InceptionV3", "dim": "2048", "classifier": "AdaBoost", "accuracy": "37.50", "precision": "45.14", "recall": "37.50", "f1": "35.22"},
        {"extractor": "InceptionV3", "dim": "2048", "classifier": "Gradient Boosting", "accuracy": "52.50", "precision": "58.67", "recall": "52.50", "f1": "51.84"},
        {"extractor": "ViT", "dim": "768", "classifier": "SVM", "accuracy": "90.00", "precision": "91.50", "recall": "90.00", "f1": "89.88"},
        {"extractor": "ViT", "dim": "768", "classifier": "XGBoost", "accuracy": "67.50", "precision": "69.05", "recall": "67.50", "f1": "67.27"},
        {"extractor": "ViT", "dim": "768", "classifier": "AdaBoost", "accuracy": "50.00", "precision": "50.62", "recall": "50.00", "f1": "46.39"},
        {"extractor": "ViT", "dim": "768", "classifier": "Gradient Boosting", "accuracy": "70.00", "precision": "80.33", "recall": "70.00", "f1": "71.52"}
    ]
    return render_template("research_benchmarks.html", results_table=results_table)

if __name__ == "__main__":
    print("\n=======================================================")
    print("SMART ATTENDANCE PORTAL - PRODUCTION SERVER")
    print("Serving on http://127.0.0.1:5000")
    print("Default Admin Login: username='admin', password='admin123'")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

