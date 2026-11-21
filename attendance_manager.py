import os
import csv
import datetime
import sqlite3
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_attendance.db")

class AttendanceManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        pass

    def _get_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def get_student_by_identity(self, identity_name):
        """
        Finds student record matching class name (e.g. 'Alejandro_Toledo' or 'Alejandro Toledo').
        """
        conn = self._get_db()
        cursor = conn.cursor()
        sanitized = identity_name.replace("_", " ")
        cursor.execute("""
            SELECT * FROM students 
            WHERE full_name LIKE ? OR id = ? OR replace(full_name, ' ', '_') = ?
            LIMIT 1
        """, (f"%{sanitized}%", identity_name, identity_name))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def mark_attendance(self, identity_name, confidence, subject="Machine Learning Systems", session_name="Regular Lecture"):
        """
        Records attendance for recognized student.
        Prevents duplicate entry for same student, date, and subject.
        Returns: (success: bool, status_message: str, record_data: dict)
        """
        if identity_name == "Unknown":
            return False, "Unknown face rejected. Attendance not marked.", None

        student = self.get_student_by_identity(identity_name)
        if not student:
            # Fallback if student not in DB yet
            student_id = f"ROLL-{identity_name.upper()[:6]}"
            full_name = identity_name.replace("_", " ")
            department = "CSE"
            academic_year = "4th Year"
            section = "Section A"
        else:
            student_id = student["id"]
            full_name = student["full_name"]
            department = student["department"]
            academic_year = student["academic_year"]
            section = student["section"]

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn = self._get_db()
        cursor = conn.cursor()

        # Check duplicate attendance for today
        cursor.execute("""
            SELECT id, attendance_time FROM attendance 
            WHERE student_id = ? AND attendance_date = ? AND subject = ?
        """, (student_id, date_str, subject))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            return False, f"Already Marked: {full_name} ({student_id}) marked today at {existing['attendance_time']}", {
                "student_id": student_id,
                "student_name": full_name,
                "date": date_str,
                "time": existing["attendance_time"],
                "subject": subject,
                "status": "Already Marked"
            }

        # Insert new record
        cursor.execute("""
            INSERT INTO attendance 
            (student_id, student_name, department, subject, academic_year, section, attendance_date, attendance_time, confidence, status, session_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Present', ?)
        """, (student_id, full_name, department, subject, academic_year, section, date_str, time_str, confidence, session_name))
        conn.commit()
        conn.close()

        # Append to Daily CSV
        csv_file = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
        file_exists = os.path.exists(csv_file)
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Student ID", "Student Name", "Department", "Academic Year", "Section", "Subject", "Date", "Time", "Confidence (%)", "Status"])
            writer.writerow([student_id, full_name, department, academic_year, section, subject, date_str, time_str, f"{confidence:.1f}", "Present"])

        record = {
            "student_id": student_id,
            "student_name": full_name,
            "department": department,
            "academic_year": academic_year,
            "section": section,
            "subject": subject,
            "date": date_str,
            "time": time_str,
            "confidence": confidence,
            "status": "Present"
        }
        return True, f"Attendance Marked: {full_name} ({student_id}) at {time_str}", record

    def get_attendance_records(self, date_filter=None, department_filter=None, subject_filter=None):
        conn = self._get_db()
        cursor = conn.cursor()
        query = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if date_filter:
            query += " AND attendance_date = ?"
            params.append(date_filter)
        if department_filter:
            query += " AND department = ?"
            params.append(department_filter)
        if subject_filter:
            query += " AND subject = ?"
            params.append(subject_filter)

        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def export_attendance_excel(self, date_filter=None, department_filter=None, subject_filter=None):
        """
        Exports attendance logs to formatted Excel file with summary metrics.
        Returns: file path to generated Excel file.
        """
        records = self.get_attendance_records(date_filter, department_filter, subject_filter)
        wb = Workbook()
        
        # Sheet 1: Detailed Logs
        ws_logs = wb.active
        ws_logs.title = "Attendance Logs"

        # Styling
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        title_font = Font(name="Segoe UI", size=14, bold=True, color="1B365D")
        border_thin = Border(left=Side(style='thin', color='DDDDDD'),
                             right=Side(style='thin', color='DDDDDD'),
                             top=Side(style='thin', color='DDDDDD'),
                             bottom=Side(style='thin', color='DDDDDD'))

        # Title
        ws_logs.append(["Smart Attendance Portal - Official Attendance Report"])
        ws_logs["A1"].font = title_font
        ws_logs.append([f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        ws_logs.append([])

        # Table Headers
        headers = ["Sl No", "Student ID", "Student Name", "Department", "Academic Year", "Section", "Subject", "Date", "Time", "Confidence (%)", "Status"]
        ws_logs.append(headers)
        
        header_row_idx = 4
        for col_idx in range(1, len(headers) + 1):
            cell = ws_logs.cell(row=header_row_idx, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for idx, r in enumerate(records, start=1):
            row_data = [
                idx,
                r.get("student_id", ""),
                r.get("student_name", ""),
                r.get("department", ""),
                r.get("academic_year", ""),
                r.get("section", ""),
                r.get("subject", ""),
                r.get("attendance_date", ""),
                r.get("attendance_time", ""),
                f"{r.get('confidence', 0):.1f}%",
                r.get("status", "Present")
            ]
            ws_logs.append(row_data)
            current_row = ws_logs.max_row
            for col_idx in range(1, len(headers) + 1):
                cell = ws_logs.cell(row=current_row, column=col_idx)
                cell.border = border_thin
                if col_idx in [1, 8, 9, 10, 11]:
                    cell.alignment = Alignment(horizontal="center")

        # Auto-fit columns
        for col in ws_logs.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            ws_logs.column_dimensions[col_letter].width = max(max_len + 3, 12)

        # Save workbook
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"Attendance_Report_{timestamp}.xlsx"
        export_path = os.path.join(ATTENDANCE_DIR, export_filename)
        wb.save(export_path)
        return export_path

