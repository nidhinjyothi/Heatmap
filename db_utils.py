"""
DB Utility Script — Manage test/dev data in AWS RDS.

"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import config


def get_conn():
    return psycopg2.connect(**config.DB_CONFIG)


# ── Read Operations ───────────────────────────────────────────────────────────

def show_latest(n=10):
    """Show the latest N rows from both tables."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print(f"\n── Latest {n} rows in sapien_vision.heatmap_data ──")
    cur.execute(f"""
        SELECT heatmap_datapoint_id, store_id, region_id, recorded_at, value, created_at
        FROM sapien_vision.heatmap_data
        ORDER BY heatmap_datapoint_id DESC
        LIMIT {n}
    """)
    rows = cur.fetchall()
    for r in rows:
        print(f"  id={r['heatmap_datapoint_id']}  store={r['store_id']}  "
              f"region={r['region_id']}  recorded={r['recorded_at']}  "
              f"value={r['value']}  created={r['created_at']}")

    print(f"\n── Latest {n} rows in sapien_vision.heatmap_evidences ──")
    cur.execute(f"""
        SELECT id, video_name, timestamp, frame_path
        FROM sapien_vision.heatmap_evidences
        ORDER BY id DESC
        LIMIT {n}
    """)
    rows = cur.fetchall()
    for r in rows:
        print(f"  id={r['id']}  video={r['video_name']}  ts={r['timestamp']}  path={r['frame_path']}")

    conn.close()


def count_rows():
    """Show total row counts."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sapien_vision.heatmap_data")
    hd_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sapien_vision.heatmap_evidences")
    he_count = cur.fetchone()[0]
    conn.close()
    print(f"\n  heatmap_data      : {hd_count} rows")
    print(f"  heatmap_evidences : {he_count} rows")


# ── Delete Operations ─────────────────────────────────────────────────────────

def delete_after_id():
    """Delete all heatmap_data rows with id > threshold (keep original 95)."""
    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("SELECT MAX(heatmap_datapoint_id) FROM sapien_vision.heatmap_data")
    max_id = cur.fetchone()[0]
    print(f"\n  Current max heatmap_datapoint_id: {max_id}")
    threshold = input("  Delete all rows with id > (enter id, e.g. 95): ").strip()

    try:
        threshold = int(threshold)
    except ValueError:
        print("  Invalid input. Cancelled.")
        conn.rollback()
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM sapien_vision.heatmap_data WHERE heatmap_datapoint_id > %s", (threshold,))
    count = cur.fetchone()[0]
    confirm = input(f"  Will delete {count} rows from heatmap_data. Confirm? (yes/no): ").strip().lower()

    if confirm == "yes":
        cur.execute("DELETE FROM sapien_vision.heatmap_data WHERE heatmap_datapoint_id > %s", (threshold,))
        conn.commit()
        print(f"Deleted {count} rows from heatmap_data.")
    else:
        conn.rollback()
        print("  Cancelled.")
    conn.close()


def delete_by_date_range():
    """Delete rows from both tables between two timestamps."""
    print("\n  Enter date range to delete (format: YYYY-MM-DD HH:MM:SS)")
    start = input("  Start: ").strip()
    end   = input("  End:   ").strip()

    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM sapien_vision.heatmap_data
        WHERE recorded_at BETWEEN %s AND %s
    """, (start, end))
    hd_count = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM sapien_vision.heatmap_evidences
        WHERE timestamp BETWEEN %s AND %s
    """, (start, end))
    he_count = cur.fetchone()[0]

    print(f"\n  Will delete:")
    print(f"    heatmap_data      : {hd_count} rows")
    print(f"    heatmap_evidences : {he_count} rows")
    confirm = input("  Confirm? (yes/no): ").strip().lower()

    if confirm == "yes":
        cur.execute("DELETE FROM sapien_vision.heatmap_data WHERE recorded_at BETWEEN %s AND %s", (start, end))
        cur.execute("DELETE FROM sapien_vision.heatmap_evidences WHERE timestamp BETWEEN %s AND %s", (start, end))
        conn.commit()
        print(f"Deleted {hd_count} + {he_count} rows.")
    else:
        conn.rollback()
        print("  Cancelled.")
    conn.close()


def delete_by_video_name():
    """Delete all rows for a specific video/camera name."""
    name = input("\n  Enter video_name to delete (e.g. Main_Store_Cam_01): ").strip()
    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    # heatmap_data doesn't have video_name — filter by created_at range instead is done by date range above
    # heatmap_evidences has video_name directly
    cur.execute("SELECT COUNT(*) FROM sapien_vision.heatmap_evidences WHERE video_name = %s", (name,))
    he_count = cur.fetchone()[0]

    print(f"\n  Will delete from heatmap_evidences: {he_count} rows for '{name}'")
    confirm = input("  Confirm? (yes/no): ").strip().lower()

    if confirm == "yes":
        cur.execute("DELETE FROM sapien_vision.heatmap_evidences WHERE video_name = %s", (name,))
        conn.commit()
        print(f"Deleted {he_count} rows.")
    else:
        conn.rollback()
        print("  Cancelled.")
    conn.close()


# ── Menu ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════╗")
    print("║   Sapien Vision — DB Utility     ║")
    print("╚══════════════════════════════════╝")

    while True:
        print("\n  1. Show row counts")
        print("  2. Show latest rows (preview)")
        print("  3. Delete rows with id > threshold  (keep original data)")
        print("  4. Delete rows by date/time range")
        print("  5. Delete heatmap_evidences by video name")
        print("  0. Exit")

        choice = input("\n  Choice: ").strip()

        if   choice == "1": count_rows()
        elif choice == "2":
            n = input("  How many rows to preview? [10]: ").strip()
            show_latest(int(n) if n else 10)
        elif choice == "3": delete_after_id()
        elif choice == "4": delete_by_date_range()
        elif choice == "5": delete_by_video_name()
        elif choice == "0":
            print("  Bye!")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
