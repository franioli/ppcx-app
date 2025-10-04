#!/usr/bin/env bash
# ...existing code...
# Combined backup: Postgres DB + DIC data volume
# - Rotates old backups by retention days
# - Validates backup files (non-empty, tar readable)
# - Sends email alert on failure if mail/mailx/sendmail available

set -euo pipefail
IFS=$'\n\t'

# ---------- Config ----------
# Database
DB_HOST="${DB_HOST:-150.145.51.193}"
DB_PORT="${DB_PORT:-5434}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-planpincieux}"
DB_PASSWORD_FILE="${DB_PASSWORD_FILE:-$HOME/secrets/db_password}"

# DIC data volume (Docker named volume)
DIC_VOLUME="${DIC_VOLUME:-ppcx-app_dic_data}"

# Backup locations
BACKUP_ROOT="${BACKUP_ROOT:-/home/fioli/storage/francesco/ppcx_db/backups}"
DB_BACKUP_DIR="${DB_BACKUP_DIR:-$BACKUP_ROOT/db}"
DIC_BACKUP_DIR="${DIC_BACKUP_DIR:-$BACKUP_ROOT/dic_data}"

# Retention in days
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Email for alerts
ALERT_EMAIL="${ALERT_EMAIL:-francescoioli@cnr.it}"   # set empty to disable

# ---------- Helpers ----------
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

send_alert() {
  local subject="$1"; shift
  local body="$*"
  if [[ -z "${ALERT_EMAIL:-}" ]]; then
    log "WARN: ALERT_EMAIL not set; skipping email. Subject: $subject"
    return
  fi
  if command -v mail >/dev/null 2>&1; then
    printf '%s\n' "$body" | mail -s "$subject" "$ALERT_EMAIL"
  elif command -v mailx >/dev/null 2>&1; then
    printf '%s\n' "$body" | mailx -s "$subject" "$ALERT_EMAIL"
  elif [[ -x /usr/sbin/sendmail ]]; then
    printf 'To: %s\nSubject: %s\n\n%s\n' "$ALERT_EMAIL" "$subject" "$body" | /usr/sbin/sendmail -t
  else
    log "WARN: No mail, mailx or sendmail found; cannot send email alert."
  fi
}

require_cmd() {
  for c in "$@"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      log "ERROR: Required command not found: $c"
      send_alert "[PPCX Backup] PPCX App Backup Failed. Missing command: $c" "Required command '$c' is not installed on $(hostname)."
      exit 1
    fi
  done
}

# ---------- Pre-flight ----------
# ensure script is run with bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  log "ERROR: This script requires bash. Run as: bash $0"
  exit 2
fi

require_cmd docker pg_dump date find tar

mkdir -p "$DB_BACKUP_DIR" "$DIC_BACKUP_DIR"

if [[ ! -r "$DB_PASSWORD_FILE" ]]; then
  log "ERROR: DB password file not readable: $DB_PASSWORD_FILE"
  send_alert "[PPCX Backup] PPCX App Backup Faild. DB password missing" "Password file not readable: $DB_PASSWORD_FILE"
  exit 1
fi

TIMESTAMP="$(date +%F_%H%M%S)"
DB_BACKUP_FILE="$DB_BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.dump"
DIC_BACKUP_FILE="$DIC_BACKUP_DIR/dic_data_${TIMESTAMP}.tar.gz"

OVERALL_OK=0
DETAILS=()

# ---------- DB Backup ----------
log "Starting DB backup: $DB_NAME -> $DB_BACKUP_FILE"
PGPASSWORD="$(cat "$DB_PASSWORD_FILE")"
export PGPASSWORD
if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -Fc -f "$DB_BACKUP_FILE"; then
  if [[ -s "$DB_BACKUP_FILE" ]]; then
    log "DB backup OK: $DB_BACKUP_FILE"
  else
    log "ERROR: DB backup file is empty: $DB_BACKUP_FILE"
    OVERALL_OK=1
    DETAILS+=("DB backup file is empty: $DB_BACKUP_FILE")
  fi
else
  log "ERROR: pg_dump failed"
  OVERALL_OK=1
  DETAILS+=("pg_dump failed for $DB_NAME@$DB_HOST:$DB_PORT")
fi
unset PGPASSWORD

# ---------- DIC Volume Backup ----------
log "Starting DIC volume backup: $DIC_VOLUME -> $DIC_BACKUP_FILE"
FILE_BN="$(basename "$DIC_BACKUP_FILE")"
if docker run --rm \
  -v "${DIC_VOLUME}":/data:ro \
  -v "${DIC_BACKUP_DIR}":/backup \
  alpine:3.20 sh -c "tar czf /backup/${FILE_BN} -C /data ."; then

  if [[ -s "$DIC_BACKUP_FILE" ]]; then
    # Validate tar is readable and not empty of entries
    if tar -tzf "$DIC_BACKUP_FILE" >/dev/null 2>&1 && [[ -n "$(tar -tzf "$DIC_BACKUP_FILE" 2>/dev/null | head -n1)" ]]; then
      log "DIC backup OK: $DIC_BACKUP_FILE"
    else
      log "ERROR: DIC backup archive appears invalid or empty: $DIC_BACKUP_FILE"
      OVERALL_OK=1
      DETAILS+=("DIC backup invalid/empty: $DIC_BACKUP_FILE")
    fi
  else
    log "ERROR: DIC backup file is empty: $DIC_BACKUP_FILE"
    OVERALL_OK=1
    DETAILS+=("DIC backup file is empty: $DIC_BACKUP_FILE")
  fi
else
  log "ERROR: Docker tar for DIC volume failed"
  OVERALL_OK=1
  DETAILS+=("Docker tar failed for volume $DIC_VOLUME")
fi

# ---------- Rotation ----------
log "Rotating backups older than $RETENTION_DAYS days"
find "$DB_BACKUP_DIR"  -type f -name "${DB_NAME}_*.dump" -mtime +"$RETENTION_DAYS" -print -delete || true
find "$DIC_BACKUP_DIR" -type f -name "dic_data_*.tar.gz" -mtime +"$RETENTION_DAYS" -print -delete || true

# ---------- Result ----------
if [[ "$OVERALL_OK" -eq 0 ]]; then
  log "Backup completed successfully."
else
  SUBJECT="[PPCX Backup] FAILURE on $(hostname) at $(date '+%F %T')"
  BODY="One or more backup steps failed.

DB: $DB_NAME@$DB_HOST:$DB_PORT
DB file: $DB_BACKUP_FILE
DIC volume: $DIC_VOLUME
DIC file: $DIC_BACKUP_FILE

Issues:
- ${DETAILS[*]}
"
  log "Backup completed with errors."
  send_alert "$SUBJECT" "$BODY"
  exit 1
fi