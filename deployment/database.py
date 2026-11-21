import os
import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_attendance.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Admins Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT,
        role TEXT DEFAULT 'admin',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Departments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL
    )
    """)
    
    # 3. Academic Years Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS academic_years (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    """)
    
    # 4. Sections Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    """)
    
    # 5. Subjects Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL
    )
    """)
    
    # 6. Faculty Members Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faculty (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faculty_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        email TEXT,
        phone TEXT
    )
    """)
    
    # 7. Students Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id TEXT PRIMARY KEY, -- Student ID / Roll Number
        full_name TEXT NOT NULL,
        department TEXT NOT NULL,
        academic_year TEXT NOT NULL,
        section TEXT NOT NULL,
        email TEXT,
        parent_email TEXT,
        photo_path TEXT,
        is_enrolled INTEGER DEFAULT 0,
        enrolled_images_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 8. Attendance Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        department TEXT NOT NULL,
        subject TEXT NOT NULL,
        academic_year TEXT,
        section TEXT,
        attendance_date TEXT NOT NULL, -- YYYY-MM-DD
        attendance_time TEXT NOT NULL, -- HH:MM:SS
        confidence REAL NOT NULL,
        status TEXT DEFAULT 'Present',
        session_name TEXT DEFAULT 'Regular Lecture',
        UNIQUE(student_id, attendance_date, subject) -- Prevent duplicate attendance for same subject on same date
    )
    """)
    
    conn.commit()
    
    # Seed default Admin user if none exists
    cursor.execute("SELECT id FROM admins WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_pass = generate_password_hash("admin123")
        cursor.execute("""
        INSERT INTO admins (username, password_hash, full_name, email, role)
        VALUES ('admin', ?, 'System Administrator', 'admin@smartattendance.edu', 'superadmin')
        """, (admin_pass,))
        print("Default admin account created: username='admin', password='admin123'")
        
    # Seed standard Departments
    default_departments = [
        ("CSE", "Computer Science & Engineering"),
        ("IT", "Information Technology"),
        ("ECE", "Electronics & Communication"),
        ("EEE", "Electrical & Electronics"),
        ("MECH", "Mechanical Engineering"),
        ("CIVIL", "Civil Engineering"),
        ("AI_DS", "Artificial Intelligence & Data Science")
    ]
    for code, name in default_departments:
        cursor.execute("INSERT OR IGNORE INTO departments (code, name) VALUES (?, ?)", (code, name))
        
    # Seed Academic Years
    default_years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
    for y in default_years:
        cursor.execute("INSERT OR IGNORE INTO academic_years (name) VALUES (?)", (y,))
        
    # Seed Sections
    default_sections = ["Section A", "Section B", "Section C", "Section D"]
    for s in default_sections:
        cursor.execute("INSERT OR IGNORE INTO sections (name) VALUES (?)", (s,))
        
    # Seed Subjects
    default_subjects = [
        ("CS801", "Deep Learning & Vision", "CSE"),
        ("CS802", "Machine Learning Systems", "CSE"),
        ("CS803", "Cloud Computing & DevOps", "CSE"),
        ("IT701", "Information Security", "IT"),
        ("EC601", "Digital Signal Processing", "ECE")
    ]
    for code, name, dept in default_subjects:
        cursor.execute("INSERT OR IGNORE INTO subjects (code, name, department) VALUES (?, ?, ?)", (code, name, dept))
        
    # Seed Faculty Members
    default_faculty = [
        ("FAC001", "Dr. Alan Turing", "CSE", "alan.turing@smartattendance.edu", "9876543210"),
        ("FAC002", "Prof. Ada Lovelace", "CSE", "ada.lovelace@smartattendance.edu", "9876543211"),
        ("FAC003", "Dr. Claude Shannon", "IT", "claude.shannon@smartattendance.edu", "9876543212")
    ]
    for fid, name, dept, email, phone in default_faculty:
        cursor.execute("INSERT OR IGNORE INTO faculty (faculty_id, name, department, email, phone) VALUES (?, ?, ?, ?, ?)", (fid, name, dept, email, phone))
        
    # Pre-seed students from face_dataset if they exist
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "face_dataset")
    if os.path.exists(dataset_dir):
        classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        for idx, student_name in enumerate(classes, start=101):
            sid = f"ROLL-{idx}"
            display_name = student_name.replace("_", " ")
            cursor.execute("SELECT id FROM students WHERE id = ? OR full_name = ?", (sid, display_name))
            if not cursor.fetchone():
                folder = os.path.join(dataset_dir, student_name)
                img_count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                cursor.execute("""
                INSERT INTO students (id, full_name, department, academic_year, section, email, parent_email, is_enrolled, enrolled_images_count)
                VALUES (?, ?, 'CSE', '4th Year', 'Section A', ?, ?, 1, ?)
                """, (sid, display_name, f"{student_name.lower()}@student.edu", f"parent.{student_name.lower()}@gmail.com", img_count))
                
    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DB_PATH)

if __name__ == "__main__":
    init_db()

