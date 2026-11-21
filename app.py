import os
import sys

# Critical: Set thread limits and cache paths BEFORE numpy, cv2, and torch load
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if "HF_HOME" not in os.environ and os.path.exists("S:\\hf_cache"):
    os.environ["HF_HOME"] = "S:\\hf_cache"

import io
import datetime
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Smart Attendance Portal | ViT + SVM",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")
if DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, DEPLOYMENT_DIR)

# Import engine modules
from deployment.database import init_db, get_connection
from deployment.face_engine import FaceRecognitionEngine
from deployment.attendance_manager import AttendanceManager

# Initialize database
init_db()

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-badge-present {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .status-badge-duplicate {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .status-badge-unknown {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_face_engine():
    return FaceRecognitionEngine.get_instance()

@st.cache_resource
def get_attendance_manager():
    return AttendanceManager.get_instance()

engine = get_face_engine()
attendance_mgr = get_attendance_manager()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/facial-recognition.png", width=72)
    st.title("Smart Attendance")
    st.caption("Deployment Pipeline: **ViT + SVM**")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "📊 Admin Dashboard",
            "📷 Biometric Attendance (Webcam)",
            "👤 Student Registration & Enrollment",
            "📋 Attendance History & Export",
            "🔬 Research Model Benchmarks"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Deployment Spec")
    st.info(
        "• **Feature Extractor**: Vision Transformer (`vit-base-patch16-224-in21k`)\n"
        "• **Classifier**: SVM (RBF Kernel)\n"
        "• **Accuracy**: **90.00%** | **F1**: **89.88%**\n"
        "• **Rejection Threshold**: 65% Confidence\n"
        "• **Duplicate Prevention**: Enabled"
    )

# ==========================================
# 1. ADMIN DASHBOARD
# ==========================================
if menu == "📊 Admin Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 28px;">🎓 Smart Attendance Administration Portal</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9;">Automated Biometric Recognition Powered by Vision Transformer (ViT) & Support Vector Machine</p>
    </div>
    """, unsafe_allow_html=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students WHERE is_enrolled = 1")
    enrolled_students = cursor.fetchone()[0]

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date = ?", (today_str,))
    today_attendance_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM faculty")
    total_faculty = cursor.fetchone()[0]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_students}</div>
            <div class="metric-label">Total Registered Students</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #16a34a;">{enrolled_students}</div>
            <div class="metric-label">Biometrically Enrolled</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #2563eb;">{today_attendance_count}</div>
            <div class="metric-label">Today's Attendance ({today_str})</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #9333ea;">90.00%</div>
            <div class="metric-label">ViT+SVM Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Department breakdown & Recent Activity
    col_chart, col_recent = st.columns([1, 1])

    with col_chart:
        st.subheader("Department Enrollment Distribution")
        cursor.execute("SELECT department, COUNT(*) as count FROM students GROUP BY department")
        dept_data = cursor.fetchall()
        if dept_data:
            df_dept = pd.DataFrame(dept_data, columns=["Department", "Students"])
            st.bar_chart(df_dept.set_index("Department"), color="#2563eb")
        else:
            st.info("No student department data available.")

    with col_recent:
        st.subheader("Recent Attendance Activity Today")
        cursor.execute("""
            SELECT student_id, student_name, subject, attendance_time, confidence, status
            FROM attendance
            WHERE attendance_date = ?
            ORDER BY id DESC LIMIT 5
        """, (today_str,))
        recent_rows = cursor.fetchall()
        if recent_rows:
            df_recent = pd.DataFrame(recent_rows, columns=["ID", "Name", "Subject", "Time", "Conf (%)", "Status"])
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance recorded yet today. Use the Webcam Attendance module to mark attendance.")

    conn.close()

# ==========================================
# 2. BIOMETRIC ATTENDANCE (WEBCAM)
# ==========================================
elif menu == "📷 Biometric Attendance (Webcam)":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 28px;">📷 Live Biometric Attendance Recognition</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9;">Accepts live camera input, extracts 768-d ViT embeddings, and classifies student identity via SVM</p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM subjects ORDER BY name ASC")
        subjects = [r[0] for r in cur.fetchall()]
        conn.close()
        if not subjects:
            subjects = ["Machine Learning Systems", "Computer Vision Lab", "Neural Networks", "Data Structures"]
        selected_subject = st.selectbox("Current Subject / Course", subjects)

    with ctrl_col2:
        selected_session = st.selectbox("Academic Session", ["Regular Lecture", "Lab Practical", "Tutorial", "Seminar"])

    with ctrl_col3:
        threshold = st.slider("Confidence Rejection Threshold (%)", min_value=50, max_value=90, value=65, step=1)
        st.caption("Identities below this confidence threshold are rejected as 'Unknown'.")

    st.markdown("---")

    input_mode = st.radio("Capture Method", ["📸 Live Webcam Capture", "📁 Upload Face Image"], horizontal=True)

    img_input = None
    if input_mode == "📸 Live Webcam Capture":
        camera_img = st.camera_input("Take a photo of the student facing the camera:")
        if camera_img is not None:
            img_input = Image.open(camera_img)
    else:
        uploaded_file = st.file_uploader("Upload an image containing the student's face", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_input = Image.open(uploaded_file)

    if img_input is not None:
        # Convert PIL to BGR OpenCV format
        rgb_arr = np.array(img_input.convert("RGB"))
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting face & extracting Vision Transformer (ViT) embeddings..."):
            # Detect faces
            faces = engine.detect_faces(bgr_arr)
            annotated_bgr = bgr_arr.copy()

            if len(faces) == 0:
                st.warning("⚠️ No face detected in the frame. Please look directly into the camera with proper lighting.")
                st.image(img_input, caption="Captured Frame", width=360)
            else:
                for (x, y, w, h) in faces:
                    pad_w = int(w * 0.1)
                    pad_h = int(h * 0.1)
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(bgr_arr.shape[1], x + w + pad_w)
                    y2 = min(bgr_arr.shape[0], y + h + pad_h)
                    face_crop = bgr_arr[y1:y2, x1:x2]

                    # Predict with ViT + SVM
                    pred_res = engine.predict_identity(face_crop, threshold=threshold / 100.0)
                    identity = pred_res["identity"]
                    conf = pred_res["confidence"]
                    is_known = pred_res["is_known"]

                    # Draw bounding box
                    box_color = (0, 200, 0) if is_known else (0, 0, 255)
                    cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), box_color, 2)
                    label_text = f"{identity.replace('_', ' ')} ({conf}%)"
                    cv2.putText(annotated_bgr, label_text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    st.image(annotated_rgb, caption="Biometric Face Analysis", use_container_width=True)

                with res_col2:
                    st.markdown("### Recognition Status")
                    if is_known:
                        st.markdown(f"""
                        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 16px; border-radius: 8px;">
                            <h3 style="color: #15803d; margin: 0;">✅ Enrolled Student Verified</h3>
                            <p style="margin: 8px 0; font-size: 18px;"><strong>Student Name:</strong> {identity.replace('_', ' ')}</p>
                            <p style="margin: 4px 0;"><strong>Model Confidence:</strong> <span style="font-weight: 700; color: #15803d;">{conf}%</span></p>
                            <p style="margin: 4px 0;"><strong>Pipeline:</strong> Vision Transformer (ViT-Base) + Support Vector Machine (RBF)</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Mark Attendance
                        success, message, rec = attendance_mgr.mark_attendance(
                            identity, confidence=conf, subject=selected_subject, session_name=selected_session
                        )

                        if success:
                            st.success(f"🎉 **{message}**")
                            st.balloons()
                        else:
                            st.warning(f"⚠️ **{message}**")
                    else:
                        st.markdown(f"""
                        <div style="background-color: #fef2f2; border: 1px solid #fecaca; padding: 16px; border-radius: 8px;">
                            <h3 style="color: #b91c1c; margin: 0;">🚫 Unknown Person Rejection</h3>
                            <p style="margin: 8px 0; font-size: 16px;">The captured face does not match any enrolled student in the database.</p>
                            <p style="margin: 4px 0;"><strong>Confidence Score:</strong> {conf}% (Threshold: {threshold}%)</p>
                            <p style="margin: 4px 0; color: #991b1b;"><strong>Action:</strong> Attendance marking was strictly rejected to preserve security.</p>
                        </div>
                        """, unsafe_allow_html=True)

# ==========================================
# 3. STUDENT REGISTRATION & ENROLLMENT
# ==========================================
elif menu == "👤 Student Registration & Enrollment":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 28px;">👤 Student Registration & Biometric Enrollment</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9;">Add new admissions and enroll facial biometric embeddings directly into the active ViT + SVM pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    tab_register, tab_enroll_existing, tab_view_students = st.tabs([
        "➕ Register New Student & Face",
        "📸 Enroll Face for Existing Student",
        "📋 Registered Student Directory"
    ])

    with tab_register:
        st.subheader("Step 1: Student Academic Information")
        with st.form("new_student_form"):
            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                student_id = st.text_input("Student ID / Roll Number *", placeholder="e.g. ROLL-2024-001").strip().upper()
                full_name = st.text_input("Full Name *", placeholder="e.g. Alexander Smith").strip()
                department = st.selectbox("Department *", ["CSE", "ECE", "IT", "EEE", "Mechanical", "Civil"])
                academic_year = st.selectbox("Academic Year *", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            with reg_c2:
                section = st.selectbox("Section *", ["Section A", "Section B", "Section C"])
                email = st.text_input("Student Email Address", placeholder="student@university.edu").strip()
                parent_email = st.text_input("Parent / Guardian Email", placeholder="parent@gmail.com").strip()

            st.subheader("Step 2: Biometric Face Images")
            st.info("Upload 3 to 10 face photos (or capture via webcam below) with diverse expressions/angles.")
            uploaded_photos = st.file_uploader(
                "Upload Student Face Images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
            )

            submit_btn = st.form_submit_button("Register & Train ViT+SVM Model")

            if submit_btn:
                if not student_id or not full_name:
                    st.error("Please provide both Student ID and Full Name.")
                else:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT id FROM students WHERE id = ?", (student_id,))
                    if cur.fetchone():
                        st.error(f"Student ID '{student_id}' is already registered in the system.")
                        conn.close()
                    else:
                        face_crops = []
                        if uploaded_photos:
                            for up_file in uploaded_photos:
                                pil_img = Image.open(up_file).convert("RGB")
                                bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                detected = engine.detect_faces(bgr)
                                if len(detected) > 0:
                                    x, y, w, h = detected[0]
                                    pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                                    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                                    x2, y2 = min(bgr.shape[1], x + w + pad_w), min(bgr.shape[0], y + h + pad_h)
                                    face_crops.append(bgr[y1:y2, x1:x2])
                                else:
                                    face_crops.append(bgr)

                        # Insert into database
                        cur.execute("""
                            INSERT INTO students (id, full_name, department, academic_year, section, email, parent_email, is_enrolled, enrolled_images_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (student_id, full_name, department, academic_year, section, email, parent_email, 1 if len(face_crops) > 0 else 0, len(face_crops)))
                        conn.commit()
                        conn.close()

                        # If faces uploaded, enroll into ViT + SVM
                        if len(face_crops) > 0:
                            with st.spinner("Extracting ViT 768-d embeddings and retraining SVM classifier..."):
                                success, msg = engine.enroll_student_faces(full_name, face_crops)
                                if success:
                                    st.success(f"🎉 **Registration & Face Enrollment Complete!**\n{msg}")
                                else:
                                    st.warning(f"Student registered, but enrollment returned: {msg}")
                        else:
                            st.success(f"Student '{full_name}' ({student_id}) registered. You can enroll face photos anytime under 'Enroll Face'.")

    with tab_enroll_existing:
        st.subheader("Add Face Samples for an Enrolled Student")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, full_name, department, is_enrolled, enrolled_images_count FROM students ORDER BY full_name ASC")
        all_studs = cur.fetchall()
        conn.close()

        if not all_studs:
            st.info("No students registered yet.")
        else:
            stud_dict = {f"{s['full_name']} ({s['id']}) - {s['department']}": s for s in all_studs}
            selected_student_str = st.selectbox("Select Student to Enroll Face Samples", list(stud_dict.keys()))
            selected_student = stud_dict[selected_student_str]

            st.write(f"Current Enrollment Status: **{'Enrolled' if selected_student['is_enrolled'] else 'Not Enrolled'}** ({selected_student['enrolled_images_count']} samples)")

            new_photos = st.file_uploader(
                "Upload new face photos for this student", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="existing_enroll"
            )
            enroll_btn = st.button("Extract ViT Embeddings & Update SVM Model")

            if enroll_btn and new_photos:
                face_crops = []
                for up_file in new_photos:
                    pil_img = Image.open(up_file).convert("RGB")
                    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    detected = engine.detect_faces(bgr)
                    if len(detected) > 0:
                        x, y, w, h = detected[0]
                        pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                        x2, y2 = min(bgr.shape[1], x + w + pad_w), min(bgr.shape[0], y + h + pad_h)
                        face_crops.append(bgr[y1:y2, x1:x2])
                    else:
                        face_crops.append(bgr)

                with st.spinner("Extracting ViT CLS tokens and updating SVM..."):
                    success, msg = engine.enroll_student_faces(selected_student["full_name"], face_crops)
                    if success:
                        conn = get_connection()
                        cur = conn.cursor()
                        cur.execute("UPDATE students SET is_enrolled = 1, enrolled_images_count = enrolled_images_count + ? WHERE id = ?", (len(face_crops), selected_student["id"]))
                        conn.commit()
                        conn.close()
                        st.success(f"🎉 **Biometric Update Succeeded!**\n{msg}")
                    else:
                        st.error(f"Failed to update model: {msg}")

    with tab_view_students:
        st.subheader("Current Enrolled Students")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, full_name, department, academic_year, section, email, is_enrolled, enrolled_images_count FROM students ORDER BY id ASC")
        rows = cur.fetchall()
        conn.close()
        if rows:
            df_stud = pd.DataFrame(rows, columns=["Roll / ID", "Name", "Department", "Year", "Section", "Email", "Enrolled", "Samples"])
            df_stud["Enrolled"] = df_stud["Enrolled"].apply(lambda x: "✅ Yes" if x == 1 else "❌ No")
            st.dataframe(df_stud, use_container_width=True, hide_index=True)
        else:
            st.info("No registered students found.")

# ==========================================
# 4. ATTENDANCE HISTORY & EXCEL EXPORT
# ==========================================
elif menu == "📋 Attendance History & Export":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 28px;">📋 Attendance Records & Excel Export</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9;">Filter historical attendance logs and export official formatted reports to Excel or CSV</p>
    </div>
    """, unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3 = st.columns(3)
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT attendance_date FROM attendance ORDER BY attendance_date DESC")
    available_dates = [r[0] for r in cur.fetchall()]
    today_default = datetime.datetime.now().strftime("%Y-%m-%d")
    if today_default not in available_dates:
        available_dates.insert(0, today_default)

    with filter_c1:
        selected_date = st.selectbox("Filter by Date", ["All Dates"] + available_dates)

    with filter_c2:
        cur.execute("SELECT code FROM departments ORDER BY code ASC")
        depts = [r[0] for r in cur.fetchall()]
        selected_dept = st.selectbox("Filter by Department", ["All Departments"] + depts)

    with filter_c3:
        cur.execute("SELECT DISTINCT subject FROM attendance ORDER BY subject ASC")
        subjs = [r[0] for r in cur.fetchall()]
        selected_subj = st.selectbox("Filter by Subject", ["All Subjects"] + subjs)

    query = "SELECT student_id, student_name, department, academic_year, section, subject, attendance_date, attendance_time, confidence, status, session_name FROM attendance WHERE 1=1"
    params = []

    if selected_date != "All Dates":
        query += " AND attendance_date = ?"
        params.append(selected_date)
    if selected_dept != "All Departments":
        query += " AND department = ?"
        params.append(selected_dept)
    if selected_subj != "All Subjects":
        query += " AND subject = ?"
        params.append(selected_subj)

    query += " ORDER BY id DESC"
    cur.execute(query, params)
    records = cur.fetchall()
    conn.close()

    st.markdown(f"**Found {len(records)} attendance records.**")

    if records:
        df_records = pd.DataFrame(records, columns=[
            "Student ID", "Student Name", "Department", "Academic Year", "Section", "Subject", "Date", "Time", "Confidence (%)", "Status", "Session"
        ])
        st.dataframe(df_records, use_container_width=True, hide_index=True)

        st.markdown("### Export Reports")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            excel_bytes = attendance_mgr.export_to_excel(
                date_filter=None if selected_date == "All Dates" else selected_date,
                department_filter=None if selected_dept == "All Departments" else selected_dept,
                subject_filter=None if selected_subj == "All Subjects" else selected_subj
            )
            st.download_button(
                label="📥 Download Official Formatted Excel Report (.xlsx)",
                data=excel_bytes,
                file_name=f"Attendance_Report_{selected_date}_{datetime.datetime.now().strftime('%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with exp_col2:
            csv_data = df_records.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV Log (.csv)",
                data=csv_data,
                file_name=f"Attendance_Log_{selected_date}.csv",
                mime="text/csv"
            )
    else:
        st.info("No attendance records match the selected filter criteria.")

# ==========================================
# 5. RESEARCH MODEL BENCHMARKS
# ==========================================
elif menu == "🔬 Research Model Benchmarks":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 28px;">🔬 Research Benchmarks & Faculty Instruction Compliance</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9;">Experimental comparison of all 16 Deep Learning Feature Extractors × Machine Learning Classifiers</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > **Faculty Directive:**  
    > *"For deployment use your best methods pipeline, not all the combinations."*
    """)

    # Load results CSV
    results_csv_path = os.path.join(BASE_DIR, "results", "ML_Classification_Results.csv")
    if os.path.exists(results_csv_path):
        df_bench = pd.read_csv(results_csv_path)
    else:
        # Fallback table
        bench_data = [
            {"Feature Extractor": "ViT", "Classifier": "SVM", "Accuracy (%)": 90.00, "Precision (%)": 91.50, "Recall (%)": 90.00, "F1 Score (%)": 89.88},
            {"Feature Extractor": "ViT", "Classifier": "XGBoost", "Accuracy (%)": 85.00, "Precision (%)": 87.20, "Recall (%)": 85.00, "F1 Score (%)": 84.65},
            {"Feature Extractor": "ViT", "Classifier": "Gradient Boosting", "Accuracy (%)": 82.50, "Precision (%)": 84.10, "Recall (%)": 82.50, "F1 Score (%)": 82.12},
            {"Feature Extractor": "ViT", "Classifier": "AdaBoost", "Accuracy (%)": 75.00, "Precision (%)": 76.50, "Recall (%)": 75.00, "F1 Score (%)": 74.80},
            {"Feature Extractor": "ResNet50", "Classifier": "SVM", "Accuracy (%)": 62.50, "Precision (%)": 65.00, "Recall (%)": 62.50, "F1 Score (%)": 61.57},
            {"Feature Extractor": "ResNet50", "Classifier": "XGBoost", "Accuracy (%)": 60.00, "Precision (%)": 62.10, "Recall (%)": 60.00, "F1 Score (%)": 59.40},
            {"Feature Extractor": "ResNet50", "Classifier": "Gradient Boosting", "Accuracy (%)": 57.50, "Precision (%)": 58.90, "Recall (%)": 57.50, "F1 Score (%)": 56.80},
            {"Feature Extractor": "ResNet50", "Classifier": "AdaBoost", "Accuracy (%)": 52.50, "Precision (%)": 53.40, "Recall (%)": 52.50, "F1 Score (%)": 51.20},
            {"Feature Extractor": "MobileNetV2", "Classifier": "SVM", "Accuracy (%)": 57.50, "Precision (%)": 60.00, "Recall (%)": 57.50, "F1 Score (%)": 56.40},
            {"Feature Extractor": "MobileNetV2", "Classifier": "XGBoost", "Accuracy (%)": 55.00, "Precision (%)": 56.50, "Recall (%)": 55.00, "F1 Score (%)": 54.10},
            {"Feature Extractor": "MobileNetV2", "Classifier": "Gradient Boosting", "Accuracy (%)": 52.50, "Precision (%)": 53.80, "Recall (%)": 52.50, "F1 Score (%)": 51.50},
            {"Feature Extractor": "MobileNetV2", "Classifier": "AdaBoost", "Accuracy (%)": 47.50, "Precision (%)": 48.00, "Recall (%)": 47.50, "F1 Score (%)": 46.20},
            {"Feature Extractor": "InceptionV3", "Classifier": "SVM", "Accuracy (%)": 55.00, "Precision (%)": 57.00, "Recall (%)": 55.00, "F1 Score (%)": 54.30},
            {"Feature Extractor": "InceptionV3", "Classifier": "XGBoost", "Accuracy (%)": 52.50, "Precision (%)": 54.00, "Recall (%)": 52.50, "F1 Score (%)": 51.80},
            {"Feature Extractor": "InceptionV3", "Classifier": "Gradient Boosting", "Accuracy (%)": 50.00, "Precision (%)": 51.20, "Recall (%)": 50.00, "F1 Score (%)": 49.50},
            {"Feature Extractor": "InceptionV3", "Classifier": "AdaBoost", "Accuracy (%)": 45.00, "Precision (%)": 46.00, "Recall (%)": 45.00, "F1 Score (%)": 44.10},
        ]
        df_bench = pd.DataFrame(bench_data)

    st.subheader("Comparative Evaluation of All 16 Pipelines")
    st.dataframe(df_bench.sort_values(by="F1 Score (%)", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("""
    ### Why Vision Transformer (ViT) + SVM was Selected:
    1. **Highest Macro F1-Score (89.88%)**: Far exceeds CNN baselines (ResNet50+SVM at 61.57%, MobileNetV2+SVM at 56.40%, InceptionV3+SVM at 54.30%).
    2. **Multi-Head Self-Attention**: Vision Transformer captures global spatial face context (eyes, nose, jawline relationships) without inductive convolution bias.
    3. **Robust Margin Maximization**: SVM with RBF kernel and probability calibration yields crisp decision boundaries with 65% rejection threshold.
    4. **Inference Efficiency**: The deployment application serializes pre-extracted weights and only initializes the winning ViT + SVM pipeline without loading extraneous experimental CNNs.
    """)
