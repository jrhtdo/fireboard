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
from fastapi.responses import FileResponse

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
#EXCEL_PATH = BASE_DIR / "task_data.xlsx"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
DB_PATH = os.path.join(os.path.dirname(__file__), "tasks.db")
LISTS_PATH = BASE_DIR / "data" / "list_defs.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALL_TASKS = []
COLUMNS = []   # current authoritative list of columns (kept in sync with DB)
import uuid
import time
from fastapi import Body
from fastapi.responses import StreamingResponse
SAVED_EXPORTS_FILE = BASE_DIR / "data" / "saved_exports.json"

def get_db_columns():
    """Return the list of columns for the tasks table from the DB, or [] if table missing."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info('tasks')")
        rows = cur.fetchall()
        return [r[1] for r in rows] if rows else []
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_all_tasks():
    """Populate ALL_TASKS and COLUMNS from the DB (if present)."""
    global ALL_TASKS, COLUMNS
    if not os.path.exists(DB_PATH):
        ALL_TASKS = []
        COLUMNS = []
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM tasks", conn)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            df.fillna("", inplace=True)
            ALL_TASKS = df.to_dict(orient="records")
            COLUMNS = list(df.columns)
        else:
            ALL_TASKS = []
            COLUMNS = get_db_columns()
    finally:
        if conn:
            conn.close()

#app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")




# Load data on startup
load_all_tasks()

# Saved filtered exports state (persisted to data/saved_exports.json)
SAVED_EXPORTS = {}  # id -> {"id": id, "name": name, "rows": [...], "created_at": ts}

def load_saved_exports():
    global SAVED_EXPORTS
    try:
        if SAVED_EXPORTS_FILE.exists():
            with open(SAVED_EXPORTS_FILE, "r", encoding="utf-8") as f:
                SAVED_EXPORTS = json.load(f)
        else:
            SAVED_EXPORTS = {}
    except Exception:
        SAVED_EXPORTS = {}

def save_saved_exports():
    try:
        SAVED_EXPORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVED_EXPORTS_FILE, "w", encoding="utf-8") as f:
            json.dump(SAVED_EXPORTS, f, indent=2, default=str)
    except Exception as e:
        print("Failed to persist saved exports:", e)

# load on startup
load_saved_exports()

@app.post("/api/save-filtered")
def api_save_filtered(payload: dict = Body(...)):
    """
    Save the payload.rows (list of row dicts) to saved exports.
    payload: { "name": optional string, "rows": [ {...}, {...} ] }
    Returns: { id, name, count, created_at }
    """
    rows = payload.get("rows", [])
    name = payload.get("name") or f"Saved set {len(SAVED_EXPORTS) + 1}"
    if not isinstance(rows, list) or len(rows) == 0:
        return JSONResponse({"status": "error", "message": "No rows provided"}, status_code=400)

    export_id = uuid.uuid4().hex
    created_at = int(time.time())
    SAVED_EXPORTS[export_id] = {
        "id": export_id,
        "name": name,
        "rows": rows,
        "count": len(rows),
        "created_at": created_at
    }
    save_saved_exports()
    return {"status": "success", "id": export_id, "name": name, "count": len(rows), "created_at": created_at}


@app.get("/api/saved-exports")
def api_list_saved_exports():
    """Return metadata for saved sets."""
    # return list sorted by created_at
    items = sorted(SAVED_EXPORTS.values(), key=lambda x: x["created_at"])
    return JSONResponse(content=items)

@app.get("/api/saved-exports/combined/download")
def api_download_combined_csv():
    """
    Combine all saved exports into one CSV, deduplicate rows where keys match exactly,
    and return as attachment.
    """
    try:
        import io
        # collect all rows
        all_rows = []
        for exp in SAVED_EXPORTS.values():
            all_rows.extend(exp.get("rows", []))
        if not all_rows:
            return JSONResponse({"error": "No saved exports"}, status_code=400)

        df = pd.DataFrame(all_rows)
        # drop exact duplicate rows
        df = df.drop_duplicates()
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = f"combined_saved_exports_{int(time.time())}.csv"
        return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/saved-exports/{export_id}")
def api_get_saved_export(export_id: str):
    exp = SAVED_EXPORTS.get(export_id)
    if not exp:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(content=exp)


@app.delete("/api/saved-exports/{export_id}")
def api_delete_saved_export(export_id: str):
    if export_id in SAVED_EXPORTS:
        del SAVED_EXPORTS[export_id]
        save_saved_exports()
        return {"status": "success"}
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.post("/api/saved-exports/clear")
def api_clear_saved_exports():
    SAVED_EXPORTS.clear()
    save_saved_exports()
    return {"status": "success"}


@app.get("/api/saved-exports/{export_id}/download")
def api_download_saved_export(export_id: str):
    """Return a CSV file of a single saved export (attachment)."""
    exp = SAVED_EXPORTS.get(export_id)
    if not exp:
        return JSONResponse({"error": "Not found"}, status_code=404)

    # Build DataFrame from rows (union columns)
    try:
        import io
        df = pd.DataFrame(exp["rows"])
        # ensure deterministic column order: sorted
        df = df[list(df.columns)]
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = f"{exp['name'].replace(' ','_')}_{export_id}.csv"
        return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/csv-template", response_class=HTMLResponse)
async def csv_template(request: Request):
    # If we don't yet have columns, show an empty template — the template can still display default headers
    fields = COLUMNS or []
    return templates.TemplateResponse("csv_template.html", {"request": request, "fields": fields})


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Accept CSV or Excel
    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".csv"):
            df_new = pd.read_csv(file.file, dtype=object)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df_new = pd.read_excel(file.file, dtype=object, engine="openpyxl")
        else:
            return {"status": "error", "message": "Only CSV or Excel files are allowed (.csv, .xlsx)"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read file: {str(e)}"}

    # Clean column names (trim whitespace)
    df_new.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df_new.columns}, inplace=True)

    # If DB exists, read existing table; else create empty df
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        try:
            df_existing = pd.read_sql("SELECT * FROM tasks", conn)
        except Exception:
            df_existing = pd.DataFrame()
        conn.close()
    else:
        df_existing = pd.DataFrame()

    # Ensure both dataframes have the same set of columns (union)
    all_cols = list(set(list(df_existing.columns) + list(df_new.columns)))
    for c in all_cols:
        if c not in df_existing.columns:
            df_existing[c] = None
        if c not in df_new.columns:
            df_new[c] = None

    # Combine and dedupe (basic drop_duplicates on full row)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    df_combined.drop_duplicates(inplace=True)

    # Write back to DB (replace the tasks table with the combined data)
    conn = sqlite3.connect(DB_PATH)
    df_combined.to_sql("tasks", conn, if_exists="replace", index=False)
    conn.close()

    # Refresh in-memory cache & columns
    load_all_tasks()

    return {"status": "success", "message": "File uploaded and merged successfully!"}

@app.get("/manual-entry", response_class=HTMLResponse)
async def manual_entry_form(request: Request):
    fields = COLUMNS or []
    return templates.TemplateResponse("manual-entry.html", {"request": request, "fields": fields})

# Manual Entry Submit Handler
@app.post("/manual-entry/")
async def handle_manual_entry(data: dict):
    try:
        # 'data' will contain submitted form fields; keep only non-empty keys
        df_new = pd.DataFrame([data])
        # Clean column names
        df_new.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df_new.columns}, inplace=True)

        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            try:
                df_existing = pd.read_sql("SELECT * FROM tasks", conn)
            except Exception:
                df_existing = pd.DataFrame()
            conn.close()
        else:
            df_existing = pd.DataFrame()

        # Union columns (add missing columns to either DF)
        all_cols = list(set(list(df_existing.columns) + list(df_new.columns)))
        for c in all_cols:
            if c not in df_existing.columns:
                df_existing[c] = None
            if c not in df_new.columns:
                df_new[c] = None

        df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        df_combined.drop_duplicates(inplace=True)

        conn = sqlite3.connect(DB_PATH)
        df_combined.to_sql("tasks", conn, if_exists="replace", index=False)
        conn.close()

        load_all_tasks()
        return {"status": "success", "message": "Manual entry added successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}


from dateutil import parser as dateparser

def parse_date(date_val):
    """Return a `date` or None. Handles datetime objects, Excel serials, and common string formats."""
    if date_val is None or (isinstance(date_val, str) and date_val.strip() == ""):
        return None
    # If it's already a datetime/date-like
    if isinstance(date_val, datetime):
        return date_val.date()

    # Try pandas (handles many formats and Excel datetimes)
    try:
        ts = pd.to_datetime(date_val, dayfirst=False, errors='coerce')  # dayfirst False by default; adjust if day-first data
        if not pd.isna(ts):
            return ts.date()
    except Exception:
        pass

    # Last resort: dateutil
    try:
        return dateparser.parse(str(date_val), dayfirst=False).date()
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
            "fields": COLUMNS,
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
    # Ensure updated_data keys are valid columns; add missing columns to df if necessary
    for k in updated_data.keys():
        if k not in df.columns:
            df[k] = ""

    # Update only columns present in df
    for col in df.columns:
        # If the submitted form did not include a column, keep existing value
        if col in updated_data:
            df.at[row_id, col] = updated_data.get(col, "")
    df.to_sql("tasks", conn, if_exists="replace", index=False)
    conn.close()

    load_all_tasks()

    return HTMLResponse(content="<script>alert('✅ Row updated successfully!'); window.location.href='/'</script>")

    
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


