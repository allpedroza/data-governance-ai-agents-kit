"""
Structured Logger for Data Discovery
JSON-based logging for analytics and monitoring
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class QueryLog:
    """Structured log entry for a query"""
    timestamp: str
    query: str
    query_length: int
    discovered_tables: int
    validated_tables: int
    invalid_tables: int
    latency_ms: int
    model: str
    embedding_model: str
    response_preview: str
    user: Optional[str] = None
    session_id: Optional[str] = None
    feedback: Optional[str] = None
    feedback_reason: Optional[str] = None
    retrieval_scores: Optional[Dict[str, float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SessionStats:
    """Statistics for a session"""
    total_queries: int = 0
    total_latency_ms: int = 0
    total_discovered: int = 0
    total_validated: int = 0
    total_invalid: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    errors: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class StructuredLogger:
    """
    JSON-based structured logger for Data Discovery queries

    Features:
    - JSONL format for easy analysis
    - Session tracking
    - Statistics aggregation
    - Console + file logging
    - Query performance metrics
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        log_file: str = "queries.jsonl",
        stats_file: str = "stats.json",
        console_level: int = logging.INFO,
        enable_console: bool = True
    ):
        """
        Initialize structured logger

        Args:
            log_dir: Directory for log files
            log_file: Name of query log file (JSONL format)
            stats_file: Name of statistics file (JSON format)
            console_level: Console logging level
            enable_console: Whether to log to console
        """
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_file)
        self.stats_path = os.path.join(log_dir, stats_file)

        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Session tracking
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_stats = SessionStats(start_time=datetime.now().isoformat())

        # Console logger
        self._console_logger = None
        if enable_console:
            self._setup_console_logger(console_level)

    def _setup_console_logger(self, level: int) -> None:
        """Setup console logger"""
        self._console_logger = logging.getLogger("data_discovery")
        self._console_logger.setLevel(level)
        self._console_logger.propagate = False

        if not self._console_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._console_logger.addHandler(handler)

    def log_query(
        self,
        query: str,
        discovered_tables: int,
        validated_tables: int,
        invalid_tables: int,
        latency_ms: int,
        model: str,
        embedding_model: str,
        response: str,
        user: Optional[str] = None,
        feedback: Optional[str] = None,
        feedback_reason: Optional[str] = None,
        retrieval_scores: Optional[Dict[str, float]] = None,
        error: Optional[str] = None
    ) -> QueryLog:
        """
        Log a query with all metadata

        Args:
            query: The user query
            discovered_tables: Number of tables discovered
            validated_tables: Number of tables validated
            invalid_tables: Number of invalid tables
            latency_ms: Query latency in milliseconds
            model: LLM model used
            embedding_model: Embedding model used
            response: Response text (truncated for log)
            user: User identifier
            feedback: User feedback (positive/negative)
            feedback_reason: Reason for feedback
            retrieval_scores: Retrieval score details
            error: Error message if any

        Returns:
            QueryLog entry
        """
        # Create log entry
        log_entry = QueryLog(
            timestamp=datetime.now().isoformat(),
            query=query,
            query_length=len(query),
            discovered_tables=discovered_tables,
            validated_tables=validated_tables,
            invalid_tables=invalid_tables,
            latency_ms=latency_ms,
            model=model,
            embedding_model=embedding_model,
            response_preview=response[:500] if response else "",
            user=user,
            session_id=self._session_id,
            feedback=feedback,
            feedback_reason=feedback_reason,
            retrieval_scores=retrieval_scores,
            error=error
        )

        # Write to JSONL file
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            if self._console_logger:
                self._console_logger.error(f"Failed to write log: {e}")

        # Update session stats
        self._update_session_stats(log_entry)

        # Console log
        if self._console_logger:
            status = "✓" if not error else "✗"
            self._console_logger.info(
                f"{status} Query: {query[:50]}... | "
                f"Tables: {validated_tables}/{discovered_tables} | "
                f"Latency: {latency_ms}ms"
            )

        return log_entry

    def _update_session_stats(self, entry: QueryLog) -> None:
        """Update session statistics"""
        self._session_stats.total_queries += 1
        self._session_stats.total_latency_ms += entry.latency_ms
        self._session_stats.total_discovered += entry.discovered_tables
        self._session_stats.total_validated += entry.validated_tables
        self._session_stats.total_invalid += entry.invalid_tables

        if entry.feedback == "positiva":
            self._session_stats.positive_feedback += 1
        elif entry.feedback == "negativa":
            self._session_stats.negative_feedback += 1

        if entry.error:
            self._session_stats.errors += 1

        self._session_stats.end_time = datetime.now().isoformat()

    def log_feedback(
        self,
        query: str,
        feedback: str,
        reason: Optional[str] = None,
        user: Optional[str] = None
    ) -> None:
        """
        Log user feedback separately

        Args:
            query: Original query
            feedback: Feedback type (positiva/negativa)
            reason: Feedback reason
            user: User identifier
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "feedback",
            "query": query,
            "feedback": feedback,
            "reason": reason,
            "user": user,
            "session_id": self._session_id
        }

        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            if self._console_logger:
                self._console_logger.error(f"Failed to write feedback: {e}")

        # Update stats
        if feedback == "positiva":
            self._session_stats.positive_feedback += 1
        elif feedback == "negativa":
            self._session_stats.negative_feedback += 1

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        stats = asdict(self._session_stats)

        # Calculate averages
        if self._session_stats.total_queries > 0:
            stats["avg_latency_ms"] = (
                self._session_stats.total_latency_ms /
                self._session_stats.total_queries
            )
            stats["avg_discovered"] = (
                self._session_stats.total_discovered /
                self._session_stats.total_queries
            )
            stats["avg_validated"] = (
                self._session_stats.total_validated /
                self._session_stats.total_queries
            )

            total_feedback = (
                self._session_stats.positive_feedback +
                self._session_stats.negative_feedback
            )
            if total_feedback > 0:
                stats["positive_rate"] = (
                    self._session_stats.positive_feedback / total_feedback
                )

        return stats

    def get_historical_stats(self, last_n: int = 100) -> Dict[str, Any]:
        """
        Get statistics from log file

        Args:
            last_n: Number of recent entries to analyze

        Returns:
            Statistics dictionary
        """
        logs = self._read_logs(last_n)

        if not logs:
            return {"total_queries": 0, "message": "No logs found"}

        # Filter to query entries only
        query_logs = [l for l in logs if l.get("type") != "feedback"]

        if not query_logs:
            return {"total_queries": 0, "message": "No query logs found"}

        # Calculate statistics
        latencies = [l["latency_ms"] for l in query_logs if "latency_ms" in l]
        discovered = [l["discovered_tables"] for l in query_logs if "discovered_tables" in l]
        validated = [l["validated_tables"] for l in query_logs if "validated_tables" in l]

        stats = {
            "total_queries": len(query_logs),
            "time_range": {
                "from": query_logs[0].get("timestamp") if query_logs else None,
                "to": query_logs[-1].get("timestamp") if query_logs else None
            }
        }

        if latencies:
            stats["latency"] = {
                "avg_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies)
            }

        if discovered:
            stats["tables"] = {
                "avg_discovered": sum(discovered) / len(discovered),
                "avg_validated": sum(validated) / len(validated) if validated else 0,
                "validation_rate": sum(validated) / sum(discovered) if sum(discovered) > 0 else 0
            }

        # Feedback stats
        feedbacks = [l for l in logs if l.get("type") == "feedback" or l.get("feedback")]
        positive = sum(1 for f in feedbacks if f.get("feedback") == "positiva")
        negative = sum(1 for f in feedbacks if f.get("feedback") == "negativa")

        if positive + negative > 0:
            stats["feedback"] = {
                "total": positive + negative,
                "positive": positive,
                "negative": negative,
                "positive_rate": positive / (positive + negative)
            }

        return stats

    def _read_logs(self, last_n: int) -> List[Dict[str, Any]]:
        """Read last N log entries"""
        if not os.path.exists(self.log_path):
            return []

        logs = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []

        return logs[-last_n:] if len(logs) > last_n else logs

    def save_session_stats(self) -> None:
        """Save session statistics to file"""
        stats = self.get_session_stats()
        stats["session_id"] = self._session_id

        try:
            # Load existing stats
            all_stats = []
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    all_stats = json.load(f)

            # Append current session
            all_stats.append(stats)

            # Keep last 100 sessions
            all_stats = all_stats[-100:]

            # Save
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(all_stats, f, indent=2, ensure_ascii=False)

        except Exception as e:
            if self._console_logger:
                self._console_logger.error(f"Failed to save stats: {e}")

    def info(self, message: str) -> None:
        """Log info message to console"""
        if self._console_logger:
            self._console_logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message to console"""
        if self._console_logger:
            self._console_logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message to console"""
        if self._console_logger:
            self._console_logger.error(message)

    def __del__(self):
        """Save stats on destruction"""
        try:
            self.save_session_stats()
        except Exception:
            pass
