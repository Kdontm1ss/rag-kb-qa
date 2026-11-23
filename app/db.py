import psycopg
from psycopg import Connection
from psycopg.rows import DictRow, dict_row


def connect() -> Connection[DictRow]:
    return psycopg.connect(connect_timeout=5, row_factory=dict_row)
