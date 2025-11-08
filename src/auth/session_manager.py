"""
Session management with SQLite storage for audit trail and user tracking.
"""

import sqlite3
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
import json


class SessionManager:
    """Manages user sessions with SQLite storage for audit logging."""

    # Session timeout duration
    SESSION_TIMEOUT_MINUTES = 480  # 8 hours

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize session manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or os.getenv(
            "AUTH_DB_PATH",
            "data/auth_sessions.db"
        )

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture TEXT,
                first_login_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL,
                login_count INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                ended_at TEXT,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)

        # Query audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                query_text TEXT NOT NULL,
                sources_used TEXT,
                query_timestamp TEXT NOT NULL,
                response_time_ms INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)

        # Authentication attempts log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                attempted_at TEXT NOT NULL,
                ip_address TEXT
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_user ON query_audit_log(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_audit_timestamp ON query_audit_log(query_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_attempts_email ON auth_attempts(email)")

        conn.commit()
        conn.close()

    def create_or_update_user(self, user_info: Dict[str, Any]) -> str:
        """
        Create or update user record.

        Args:
            user_info: User information from OAuth (supports both Google and Slack)

        Returns:
            user_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Handle both Google OAuth ('sub') and Slack OAuth ('id') user ID fields
        user_id = user_info.get('sub') or user_info.get('id')
        if not user_id:
            raise ValueError("User info must contain either 'sub' (Google) or 'id' (Slack) field")

        email = user_info['email']
        # Slack uses 'real_name' or 'display_name', Google uses 'name'
        name = user_info.get('name') or user_info.get('real_name') or user_info.get('display_name', '')
        # Slack uses 'image', Google uses 'picture'
        picture = user_info.get('picture') or user_info.get('image', '')
        now = datetime.utcnow().isoformat()

        try:
            # Check if user exists by user_id
            cursor.execute("SELECT login_count FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()

            if result:
                # Update existing user
                login_count = result[0] + 1
                cursor.execute("""
                    UPDATE users
                    SET email = ?, name = ?, picture = ?, last_login_at = ?, login_count = ?
                    WHERE user_id = ?
                """, (email, name, picture, now, login_count, user_id))
            else:
                # Check if email exists (user might have logged in with different OAuth provider)
                cursor.execute("SELECT user_id, login_count FROM users WHERE email = ?", (email,))
                email_result = cursor.fetchone()

                if email_result:
                    # User exists with this email but different user_id - update to new user_id
                    old_user_id = email_result[0]
                    login_count = email_result[1] + 1
                    cursor.execute("""
                        UPDATE users
                        SET user_id = ?, name = ?, picture = ?, last_login_at = ?, login_count = ?
                        WHERE email = ?
                    """, (user_id, name, picture, now, login_count, email))
                else:
                    # Create new user
                    cursor.execute("""
                        INSERT INTO users (user_id, email, name, picture, first_login_at, last_login_at, login_count)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                    """, (user_id, email, name, picture, now, now))

            conn.commit()
            return user_id

        finally:
            conn.close()

    def create_session(self, user_info: Dict[str, Any]) -> int:
        """
        Create a new session for authenticated user.

        Args:
            user_info: User information from OAuth

        Returns:
            session_id
        """
        # First, create or update user
        user_id = self.create_or_update_user(user_info)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = datetime.utcnow().isoformat()
            email = user_info['email']

            cursor.execute("""
                INSERT INTO sessions (user_id, email, started_at, last_activity_at, is_active)
                VALUES (?, ?, ?, ?, 1)
            """, (user_id, email, now, now))

            session_id = cursor.lastrowid
            conn.commit()

            return session_id

        finally:
            conn.close()

    def update_session_activity(self, session_id: int):
        """
        Update last activity timestamp for session.

        Args:
            session_id: Session ID to update
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = datetime.utcnow().isoformat()
            cursor.execute("""
                UPDATE sessions
                SET last_activity_at = ?
                WHERE session_id = ?
            """, (now, session_id))

            conn.commit()

        finally:
            conn.close()

    def end_session(self, session_id: int):
        """
        End a session.

        Args:
            session_id: Session ID to end
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = datetime.utcnow().isoformat()
            cursor.execute("""
                UPDATE sessions
                SET ended_at = ?, is_active = 0
                WHERE session_id = ?
            """, (now, session_id))

            conn.commit()

        finally:
            conn.close()

    def is_session_valid(self, session_id: int) -> bool:
        """
        Check if session is still valid (not expired).

        Args:
            session_id: Session ID to check

        Returns:
            True if session is valid, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT last_activity_at, is_active
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))

            result = cursor.fetchone()

            if not result or not result[1]:
                return False

            last_activity_str = result[0]
            last_activity = datetime.fromisoformat(last_activity_str)
            timeout_threshold = datetime.utcnow() - timedelta(minutes=self.SESSION_TIMEOUT_MINUTES)

            return last_activity > timeout_threshold

        finally:
            conn.close()

    def log_query(self, session_id: int, user_id: str, email: str, query_text: str,
                  sources_used: Optional[List[str]] = None, response_time_ms: Optional[int] = None):
        """
        Log a user query for audit trail.

        Args:
            session_id: Active session ID
            user_id: User ID
            email: User email
            query_text: The query text
            sources_used: List of data sources used (e.g., ['confluence', 'slack'])
            response_time_ms: Response time in milliseconds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = datetime.utcnow().isoformat()
            sources_json = json.dumps(sources_used) if sources_used else None

            cursor.execute("""
                INSERT INTO query_audit_log (session_id, user_id, email, query_text, sources_used, query_timestamp, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, email, query_text, sources_json, now, response_time_ms))

            conn.commit()

        finally:
            conn.close()

    def log_auth_attempt(self, email: Optional[str], success: bool, error_message: Optional[str] = None):
        """
        Log authentication attempt.

        Args:
            email: User email (if available)
            success: Whether authentication succeeded
            error_message: Error message if failed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = datetime.utcnow().isoformat()

            cursor.execute("""
                INSERT INTO auth_attempts (email, success, error_message, attempted_at)
                VALUES (?, ?, ?, ?)
            """, (email, success, error_message, now))

            conn.commit()

        finally:
            conn.close()

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Dictionary with user statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get user info
            cursor.execute("""
                SELECT email, name, first_login_at, last_login_at, login_count
                FROM users
                WHERE user_id = ?
            """, (user_id,))

            user_row = cursor.fetchone()

            if not user_row:
                return {}

            # Get query count
            cursor.execute("""
                SELECT COUNT(*)
                FROM query_audit_log
                WHERE user_id = ?
            """, (user_id,))

            query_count = cursor.fetchone()[0]

            # Get session count
            cursor.execute("""
                SELECT COUNT(*)
                FROM sessions
                WHERE user_id = ?
            """, (user_id,))

            session_count = cursor.fetchone()[0]

            return {
                'email': user_row[0],
                'name': user_row[1],
                'first_login_at': user_row[2],
                'last_login_at': user_row[3],
                'login_count': user_row[4],
                'total_queries': query_count,
                'total_sessions': session_count
            }

        finally:
            conn.close()

    def get_user_from_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user info from session ID.

        Args:
            session_id: Session ID

        Returns:
            User info dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT u.user_id, u.email, u.name, u.picture
                FROM sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.session_id = ? AND s.is_active = 1
            """, (session_id,))

            row = cursor.fetchone()

            if row:
                return {
                    'sub': row[0],
                    'email': row[1],
                    'name': row[2],
                    'picture': row[3],
                    'email_verified': True
                }

            return None

        finally:
            conn.close()

    def get_recent_queries(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent queries for a user.

        Args:
            user_id: User ID
            limit: Maximum number of queries to return

        Returns:
            List of query dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT query_text, sources_used, query_timestamp, response_time_ms
                FROM query_audit_log
                WHERE user_id = ?
                ORDER BY query_timestamp DESC
                LIMIT ?
            """, (user_id, limit))

            rows = cursor.fetchall()

            queries = []
            for row in rows:
                queries.append({
                    'query_text': row[0],
                    'sources_used': json.loads(row[1]) if row[1] else [],
                    'query_timestamp': row[2],
                    'response_time_ms': row[3]
                })

            return queries

        finally:
            conn.close()


def get_session_manager() -> SessionManager:
    """
    Get or create the global session manager instance.

    Returns:
        SessionManager instance
    """
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()

    return st.session_state.session_manager
