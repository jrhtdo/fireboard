from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import os
import json
from pathlib import Path
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
#EXCEL_PATH = BASE_DIR / "task_data.xlsx"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
DB_PATH = os.path.join(os.path.dirname(__file__), "tasks.db")
LISTS_PATH = BASE_DIR / "data" / "list_defs.json"


REQUIRED_COLUMNS = [
    "Region",
    "Address",
    "Unit No",
    "Perceived sites",
    "Scheduled Date",
    "Plan Finish Date",
    "Scheduled Time",
    "Sequence",
    "Equipment_No",
    "Audit WorkOrder",
    "Repair WorkOrder",
    "Work Description",
    "Status",
    "Assigned Person",
    "Safety Alert",
    "Post Code",
    "Note",
    "Month",
    "Assignee"
]

#app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALL_TASKS = []

def load_all_tasks():
    global ALL_TASKS
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM tasks", conn)
        conn.close()
        df.fillna("", inplace=True)
        ALL_TASKS = df.to_dict(orient="records")
    else:
        ALL_TASKS = []


# Load data on startup
load_all_tasks()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/csv-template", response_class=HTMLResponse)
async def csv_template(request: Request):
    return templates.TemplateResponse("csv_template.html", {"request": request, "fields": REQUIRED_COLUMNS})

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"status": "error", "message": "Only CSV files are allowed"}

    df_new = pd.read_csv(file.file)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_new.columns]
    if missing_cols:
        return {"status": "error", "message": f"Missing columns: {', '.join(missing_cols)}"}

    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df_existing = pd.read_sql("SELECT * FROM tasks", conn)
        conn.close()
    else:
        df_existing = pd.DataFrame(columns=REQUIRED_COLUMNS)


    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(inplace=True)
    conn = sqlite3.connect(DB_PATH)
    df_combined.to_sql("tasks", conn, if_exists="replace", index=False)
    conn.close()


    load_all_tasks()

    return {"status": "success", "message": "CSV uploaded and merged successfully!"}

@app.get("/manual-entry", response_class=HTMLResponse)
async def manual_entry_form(request: Request):
    return templates.TemplateResponse("manual-entry.html", {"request": request, "fields": REQUIRED_COLUMNS})

# Manual Entry Submit Handler
@app.post("/manual-entry/")
async def handle_manual_entry(data: dict):
    try:
        df_new = pd.DataFrame([data])
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            df_existing = pd.read_sql("SELECT * FROM tasks", conn)
            conn.close()
        else:
            df_existing = pd.DataFrame(columns=REQUIRED_COLUMNS)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(inplace=True)
        conn = sqlite3.connect(DB_PATH)
        df_combined.to_sql("tasks", conn, if_exists="replace", index=False)
        conn.close()

        return {"status": "success", "message": "Manual entry added successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

def parse_date(date_val):
    if isinstance(date_val, datetime):
        return date_val.date()
    try:
        return datetime.strptime(date_val, "%d/%m/%Y").date()
    except Exception:
        return None


@app.get("/due-count")
def get_due_count():
    today = datetime.today().date()
    threshold = today + timedelta(days=10)

    due_items = [
        task for task in ALL_TASKS
        if "Plan Finish Date" in task
        and parse_date(task["Plan Finish Date"]) is not None
        and today <= parse_date(task["Plan Finish Date"]) <= threshold
    ]
    return {"count": len(due_items)}


@app.get("/pending_tasks", response_class=HTMLResponse)
async def pending_tasks(request: Request):
    today = datetime.today().date()
    threshold = today + timedelta(days=10)

    due_items = []
    for task in ALL_TASKS:
        date_str = task.get("Plan Finish Date", "")
        parsed_date = parse_date(date_str)

        if parsed_date and today <= parsed_date <= threshold:
            due_items.append({
                "Region": task.get("Region", ""),
                "Equipment Number": task.get("Equipment_No", ""),
                "Unit No": task.get("Unit No", ""),
                "Address": task.get("Address", ""),
                "Post Code": task.get("Post Code", ""),
                "Plan Finish Date": date_str
            })

    return templates.TemplateResponse("pending_tasks.html", {
        "request": request,
        "tasks": due_items
    })

@app.get("/api/data")
def get_data():
    return JSONResponse(content=ALL_TASKS)
from fastapi import Body

@app.get("/edit/{row_id}", response_class=HTMLResponse)
async def edit_row(request: Request, row_id: int):
    try:
        row_data = ALL_TASKS[row_id]
        return templates.TemplateResponse("edit-entry.html", {
            "request": request,
            "fields": REQUIRED_COLUMNS,
            "row_data": row_data,
            "row_id": row_id
        })
    except IndexError:
        return HTMLResponse("Invalid row ID", status_code=404)
@app.post("/edit/{row_id}")
async def save_edit(request: Request, row_id: int):
    form = await request.form()
    updated_data = dict(form)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM tasks", conn)
    for col in REQUIRED_COLUMNS:
        df.at[row_id, col] = updated_data.get(col, "")
    df.to_sql("tasks", conn, if_exists="replace", index=False)
    conn.close()

    load_all_tasks()

    return HTMLResponse(content=f"<script>alert('✅ Row updated successfully!'); window.location.href='/'</script>")
from pydantic import BaseModel
from typing import List

class BulkUpdateItem(BaseModel):
    rowIndex: int
    column: str
    value: str

class BulkUpdatePayload(BaseModel):
    updates: List[BulkUpdateItem]

@app.post("/api/bulk-update")
def bulk_update(payload: BulkUpdatePayload):
    # Load into DataFrame
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM tasks", conn)
    # Apply each update
    for item in payload.updates:
        if item.column in df.columns and 0 <= item.rowIndex < len(df):
            df.at[item.rowIndex, item.column] = item.value
    # Write back
    df.to_sql("tasks", conn, if_exists="replace", index=False)
    conn.close()
    load_all_tasks()  # refresh your in-memory cache
    return {"status": "success", "message": "Bulk update applied."}
from fastapi import Body

@app.get("/api/lists")
def get_lists():
    try:
        with open(LISTS_PATH, "r") as f:
            return JSONResponse(content=json.load(f))
    except FileNotFoundError:
        return JSONResponse(content={}, status_code=200)

@app.post("/api/lists")
def save_lists(payload: dict = Body(...)):
    try:
        with open(LISTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
from io import StringIO


